// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "otsolver_trianglemesh.h"
#include "utils/BenchTimer.h"
#include <Eigen/Eigenvalues>
#include <cmath>

using namespace Eigen;


namespace otmap {

TriangularMeshTransportSolver::
TriangularMeshTransportSolver()
  : m_verbose_level(1)
{
}

TriangularMeshTransportSolver::
~TriangularMeshTransportSolver()
{
}

void
TriangularMeshTransportSolver::
adjust_density(VectorXd& density, double max_ratio)
{
  assert(density.size() == m_pb_size);

  auto f_areas = m_mesh->face_property<double>("f:areas", 0.0);
  double total_area = 0;
  for(auto f : m_mesh->faces()) {
      surface_mesh::Surface_mesh::Halfedge h = m_mesh->halfedge(f);
      const surface_mesh::Point& p0 = m_mesh->position(m_mesh->to_vertex(h));
      const surface_mesh::Point& p1 = m_mesh->position(m_mesh->to_vertex(m_mesh->next_halfedge(h)));
      const surface_mesh::Point& p2 = m_mesh->position(m_mesh->from_vertex(h));
      double area = 0.5 * std::abs((p1.head<2>() - p0.head<2>()).cross(p2.head<2>() - p0.head<2>()));
      f_areas[f] = area;
      total_area += area;
  }
  
  double I = 0;
  for(int i=0; i<density.size(); ++i) {
    I += density[i] * f_areas[surface_mesh::Surface_mesh::Face(i)];
  }
  if (I > 1e-9)
    density /= I;

  if(m_verbose_level >= 2){
    double new_I = 0;
    for(int i=0; i<density.size(); ++i) {
      new_I += density[i] * f_areas[surface_mesh::Surface_mesh::Face(i)];
    }
    std::cout << "[density]" << std::endl;
    std::cout << "  - density integral : " << new_I << std::endl;
    std::cout << "  - density stats : min = " << density.minCoeff()
                               << ", max = " << density.maxCoeff()
                               << ", ratio = " << density.maxCoeff() / density.minCoeff()
                               << std::endl;
  }
  m_mesh->remove_face_property(f_areas);
}

void
TriangularMeshTransportSolver::
init(std::shared_ptr<surface_mesh::Surface_mesh> mesh)
{
    m_mesh = mesh;
    if(m_verbose_level>=1)
      std::cout << "Init triangular solver...\n";
    
    BenchTimer timer;
    timer.start();
    this->initialize_laplacian_solver();
    timer.stop();

    if(m_verbose_level>=1)
      std::cout << " - done in " << timer.value(REAL_TIMER) << " s\n";
}

TransportMap
TriangularMeshTransportSolver::solve(ConstRefVector in_density, SolverOptions opt)
{
  if(m_verbose_level>=1)
  {
    std::cout << " Solve transport map using beta=";
    if(opt.beta==BetaOpt::Zero)               std::cout << "0";
    if(opt.beta==BetaOpt::ConjugateJacobian)  std::cout << "Conjugate-Jacobian";
    std::cout << " ;  max_iter=" << opt.max_iter;
    std::cout << " ;  threshold=" << opt.threshold;
  }

  int n = m_pb_size;

  std::shared_ptr<VectorXd> p_density = std::make_shared<VectorXd>(in_density);
  if (n != p_density->size()) {
      std::cerr << "Error: input density vector has wrong size!" << std::endl;
      return TransportMap(nullptr, nullptr, nullptr);
  }
  adjust_density(*p_density, opt.max_ratio);
  m_input_density = p_density.get();

  m_cache_1D_g0.setZero(m_mesh->n_vertices(), 2);

  VectorXd xk   = VectorXd::Zero(n);
  VectorXd xkp1 = VectorXd::Zero(n);
  VectorXd rkm1  = VectorXd::Zero(n);
  VectorXd rk    = VectorXd::Zero(n);
  VectorXd rkp1  = VectorXd::Zero(n);
  VectorXd d_hat = VectorXd::Zero(n);
  VectorXd d     = VectorXd::Zero(n);
  double beta = 0;
  double alpha = 0;

  double residual = compute_residual(xk,rkp1);

  if(m_verbose_level>=1) {
    std::cout << "  ; initial L2=" << residual
              << " Linf=" << rkp1.array().abs().maxCoeff() << "\n";
  }

  double t_linearsolve, t_beta, t_linesearch;
  int it = 0;
  BenchTimer timer;
  timer.start();
  double t_linearsolve_sum = 0., t_beta_sum = 0., t_linesearch_sum = 0.;

  while(it < opt.max_iter && residual > opt.threshold){
    if(m_verbose_level>=4) std::cout << " ===> Iteration #" << it+1 << " <===\n";
    timer.start();
    rkm1.swap(rk);
    rk.swap(rkp1);
    d_hat = m_laplacian_solver.solve(rk);
    d_hat.array() -= d_hat.mean();

    if(d_hat.norm()<=2*std::numeric_limits<double>::min())
      break;
    
    timer.stop(); t_linearsolve = timer.value(REAL_TIMER); t_linearsolve_sum += t_linearsolve; timer.start();

    if(it<1 || opt.beta==BetaOpt::Zero)
    {
      d = d_hat;
    }
    else 
    {
      beta = compute_conjugate_jacobian_beta(xk,rkm1,rk,d_hat,d,alpha);
      d = d_hat + beta*d;
    }

    timer.stop(); t_beta = timer.value(REAL_TIMER); t_beta_sum += t_beta; timer.start();
    alpha = 0;
    residual = solve_1D_problem(xk, d, rk, residual, xkp1, rkp1, &alpha);
    xk.swap(xkp1);

    timer.stop(); t_linesearch = timer.value(REAL_TIMER); t_linesearch_sum += t_linesearch;
    print_debuginfo_iteration(it, alpha, beta, d, residual, rkp1, t_linearsolve, t_beta, t_linesearch);
    ++it;
  }
  timer.stop();

  compute_vertex_gradients(xk, m_cache_residual_vtx_grads);
  auto forward_mesh = std::make_shared<surface_mesh::Surface_mesh>(*m_mesh);
  for(unsigned int j=0; j<m_cache_residual_vtx_grads.rows(); ++j)
  {
    auto& pt = forward_mesh->position(surface_mesh::Surface_mesh::Vertex(j));
    pt.x() += m_cache_residual_vtx_grads(j,0);
    pt.y() += m_cache_residual_vtx_grads(j,1);
  }

  if(m_verbose_level >= 1) {
    std::cout << " Solution:\n";
    std::cout << "  - timings: [" << "solve("
              << t_linearsolve_sum/double(it) << ") + beta("
              << t_beta_sum/double(it) << ") + linesearch("
              << t_linesearch_sum/double(it) << ")] * iters(" << it << ") = " << timer.total(REAL_TIMER) << "s\n";
    std::cout << "  - error L2=" << residual
              << "   Linf=" << rkp1.array().abs().maxCoeff() << "\n";
  }
  if(m_verbose_level >= 3) {
    VectorXd ot_cost_per_face;
    compute_transport_cost(m_cache_residual_vtx_grads,ot_cost_per_face);
    std::cout << "  - transport cost=" << ot_cost_per_face.sum() << std::endl;
  }

  return TransportMap(m_mesh, forward_mesh, p_density);
}


double
TriangularMeshTransportSolver::compute_conjugate_jacobian_beta(ConstRefVector xk, ConstRefVector rkm1, ConstRefVector rk, ConstRefVector d_hat, ConstRefVector d_prev, double alpha) const
{
  int n = m_pb_size;
  m_cache_beta_Jd.resize(n);
  m_cache_beta_rk_eps.resize(n);

  double eps = alpha/2.;
  if (eps == 0) return 0.;
  compute_residual(xk-eps*d_hat, m_cache_beta_rk_eps);

  m_cache_beta_Jd = (rk-rkm1);
  
  double denom = m_cache_beta_Jd.squaredNorm();
  if (denom < 1e-20) return 0.;

  return std::max(-1.,double(m_cache_beta_Jd.dot(m_cache_beta_rk_eps-rk)) / denom / (eps) * alpha);
}

void
TriangularMeshTransportSolver::
initialize_laplacian_solver()
{
    BenchTimer timer;

    int nf = m_mesh->n_faces();
    m_pb_size = nf;

    timer.start();
    {


        typedef Triplet<double, int> Triplet;
        std::vector<Triplet> L_entries;
        L_entries.reserve(nf * 4);
        
        std::vector<double> diag(nf, 0.0);

        for (auto e : m_mesh->edges()) {
            if (!m_mesh->is_boundary(e)) {
                surface_mesh::Surface_mesh::Face f0 = m_mesh->face(m_mesh->halfedge(e, 0));
                surface_mesh::Surface_mesh::Face f1 = m_mesh->face(m_mesh->halfedge(e, 1));
                
                double weight = m_mesh->edge_length(e);

                L_entries.push_back(Triplet(f0.idx(), f1.idx(), -weight));
                L_entries.push_back(Triplet(f1.idx(), f0.idx(), -weight));
                diag[f0.idx()] += weight;
                diag[f1.idx()] += weight;
            }
        }

        for(int i=0; i<nf; ++i)
            L_entries.push_back(Triplet(i, i, diag[i]));

        m_mat_L.resize(nf, nf);
        m_mat_L.setFromTriplets(L_entries.begin(), L_entries.end());
    }
    timer.stop();

    if (m_verbose_level >= 2)
        std::cout << "  - Laplacian matrix computed in " << timer.value(REAL_TIMER) << " s" << std::endl;

    timer.start();
    {
        m_mat_L.coeffRef(0,0) += std::abs(m_mat_L.coeffRef(0,0))*1e-4;
        m_laplacian_solver.analyzePattern(m_mat_L);
        m_laplacian_solver.factorize(m_mat_L);

        if (m_laplacian_solver.info() != Success) {
            std::cout << "Solver.Info = ";
            if (m_laplacian_solver.info() == NumericalIssue) std::cout << "NumericalIssue\n";
            else if (m_laplacian_solver.info() == NoConvergence) std::cout << "NoConvergence\n";
            else if (m_laplacian_solver.info() == InvalidInput) std::cout << "InvalidInput\n";
            else std::cout << "\n";
        }
    }
    timer.stop();

    if (m_verbose_level >= 2)
        std::cout << "  - Cholesky(Laplacian) done in " << timer.value(REAL_TIMER) << " s\n";
}

void
TriangularMeshTransportSolver::
compute_vertex_gradients(ConstRefVector psi, MatrixX2d& vtx_grads) const
{
    int nv = m_mesh->n_vertices();
    vtx_grads.resize(nv, 2);
    vtx_grads.setZero();

    auto f_centroids = m_mesh->face_property<surface_mesh::Point>("f:centroids");
    for (auto f : m_mesh->faces()) {
        surface_mesh::Point centroid(0,0);
        double n = 0;
        for (auto v : m_mesh->vertices(f)) {
            centroid += m_mesh->position(v).head<2>();
            n += 1;
        }
        if (n > 0)
          f_centroids[f] = centroid / n;
    }
    
    for (auto v : m_mesh->vertices()) {
        const Vector2d pos_v = m_mesh->position(v).head<2>();
        for (auto f : m_mesh->faces(v)) {
            const Vector2d& centroid = f_centroids[f].head<2>();
            Vector2d grad_f = (centroid - pos_v) * psi[f.idx()];
            vtx_grads.row(v.idx()) += grad_f;
        }
    }

    m_mesh->remove_face_property(f_centroids);
}

double
TriangularMeshTransportSolver::
compute_residual(ConstRefVector psi, Ref<VectorXd> out) const
{
  int nf = m_mesh->n_faces();
  out.resize(nf);

  MatrixX2d &vtx_grads(m_cache_1D_g0);
  compute_vertex_gradients(psi, vtx_grads);

  for (auto f : m_mesh->faces()) {
      surface_mesh::Surface_mesh::Halfedge h = m_mesh->halfedge(f);
      surface_mesh::Surface_mesh::Vertex v0 = m_mesh->to_vertex(h);
      surface_mesh::Surface_mesh::Vertex v1 = m_mesh->to_vertex(m_mesh->next_halfedge(h));
      surface_mesh::Surface_mesh::Vertex v2 = m_mesh->from_vertex(h);

      const Vector2d p0 = m_mesh->position(v0).head<2>();
      const Vector2d p1 = m_mesh->position(v1).head<2>();
      const Vector2d p2 = m_mesh->position(v2).head<2>();

      const Vector2d g0 = vtx_grads.row(v0.idx());
      const Vector2d g1 = vtx_grads.row(v1.idx());
      const Vector2d g2 = vtx_grads.row(v2.idx());

      const Vector2d p_prime0 = p0 + g0;
      const Vector2d p_prime1 = p1 + g1;
      const Vector2d p_prime2 = p2 + g2;

      double original_area = 0.5 * std::abs(p0.x() * (p1.y() - p2.y()) + p1.x() * (p2.y() - p0.y()) + p2.x() * (p0.y() - p1.y()));
      double transported_area = 0.5 * std::abs(p_prime0.x() * (p_prime1.y() - p_prime2.y()) + p_prime1.x() * (p_prime2.y() - p_prime0.y()) + p_prime2.x() * (p_prime0.y() - p_prime1.y()));
      
      out[f.idx()] = transported_area - original_area * (*m_input_density)[f.idx()];
  }

  return out.squaredNorm();
}

void
TriangularMeshTransportSolver::
compute_1D_problem_parameters(Ref<const VectorXd> psi, Ref<const VectorXd> dir, Ref<VectorXd> a, Ref<VectorXd> b) const
{
    int nf = m_mesh->n_faces();
    a.resize(nf);
    b.resize(nf);

    MatrixX2d &g0(m_cache_1D_g0);
    MatrixX2d &gd(m_cache_1D_gd);
    compute_vertex_gradients(dir, gd);

    for (auto f : m_mesh->faces()) {
        surface_mesh::Surface_mesh::Halfedge h = m_mesh->halfedge(f);
        surface_mesh::Surface_mesh::Vertex v0 = m_mesh->to_vertex(h);
        surface_mesh::Surface_mesh::Vertex v1 = m_mesh->to_vertex(m_mesh->next_halfedge(h));
        surface_mesh::Surface_mesh::Vertex v2 = m_mesh->from_vertex(h);

        const Vector2d p0 = m_mesh->position(v0).head<2>();
        const Vector2d p1 = m_mesh->position(v1).head<2>();
        const Vector2d p2 = m_mesh->position(v2).head<2>();

        const Vector2d p_prime0 = p0 + g0.row(v0.idx()).transpose();
        const Vector2d p_prime1 = p1 + g0.row(v1.idx()).transpose();
        const Vector2d p_prime2 = p2 + g0.row(v2.idx()).transpose();
        
        const Vector2d p_sec0 = gd.row(v0.idx()).transpose();
        const Vector2d p_sec1 = gd.row(v1.idx()).transpose();
        const Vector2d p_sec2 = gd.row(v2.idx()).transpose();

        const Vector2d dp_prime10 = p_prime1 - p_prime0;
        const Vector2d dp_prime20 = p_prime2 - p_prime0;
        const Vector2d dp_sec10 = p_sec1 - p_sec0;
        const Vector2d dp_sec20 = p_sec2 - p_sec0;

        double C1 = 0.5 * ( (dp_prime10.x() * dp_sec20.y() + dp_sec10.x() * dp_prime20.y()) - 
                              (dp_prime10.y() * dp_sec20.x() + dp_sec10.y() * dp_prime20.x()) );
        
        double C2 = 0.5 * ( (dp_sec10.x() * dp_sec20.y()) - (dp_sec10.y() * dp_sec20.x()) );

        a[f.idx()] = C2;
        b[f.idx()] = C1;
    }
}

double
TriangularMeshTransportSolver::
solve_1D_problem(ConstRefVector xk, ConstRefVector dir, ConstRefVector rk, double ek, RefVector xk1, RefVector rk1, double *palpha) const
{
  VectorXd &a(m_cache_1D_a);
  VectorXd &b(m_cache_1D_b);
  a.resize(xk.size());
  b.resize(xk.size());

  compute_1D_problem_parameters(xk, dir, a, b);

  Matrix<double,5,1> z;
  z <<  ek, 
        2.*b.dot(rk),
        b.squaredNorm()+2*a.dot(rk),
        2.*a.dot(b),
        a.squaredNorm();

  Matrix<double,4,1> w;
  w << z(1), 2.*z(2), 3.*z(3), 4.*z(4);

  Matrix3d C(3,3);
  C.setZero();
  C.bottomLeftCorner(2,2).setIdentity();
  
  if (std::abs(w(3)) < 1e-12) {
      if (palpha) *palpha = 0;
      xk1 = xk;
      rk1 = rk;
      return ek;
  }

  C.col(2) = -w.head(3)/w(3);
  EigenSolver<Matrix3d> eig(C);
  double rmin_sq = -1.;
  double alpha = 0;
  bool found = false;
  for(int k=0; k<3; ++k)
  {
    std::complex<double> root = eig.eigenvalues()(k);
    if(root.imag()==0.) {
      double t = root.real();
      double r_sq = z(0) + t*(z(1)+t*(z(2)+t*(z(3)+t*z(4))));
      if((!found) || (r_sq<rmin_sq)) {
        found = true;
        alpha = t;
        rmin_sq = r_sq;
      }
    }
  }

  if(!found) {
    alpha = 0;
    rmin_sq = z(0);
  }

  if(palpha)
    *palpha = alpha;
  xk1 = xk + alpha * dir;

  m_cache_1D_g0 += alpha * m_cache_1D_gd;
  rk1 = a*(alpha*alpha)+b*alpha+rk;
  return rmin_sq;
}

void
TriangularMeshTransportSolver::
compute_transport_cost(const MatrixX2d& vtx_grads, VectorXd &cost) const
{
    if(cost.size() != m_pb_size){
        cost.resize(m_pb_size);
        cost.setZero();
    }

    for (auto f : m_mesh->faces()) {
        surface_mesh::Surface_mesh::Halfedge h = m_mesh->halfedge(f);
        surface_mesh::Surface_mesh::Vertex v0 = m_mesh->to_vertex(h);
        surface_mesh::Surface_mesh::Vertex v1 = m_mesh->to_vertex(m_mesh->next_halfedge(h));
        surface_mesh::Surface_mesh::Vertex v2 = m_mesh->from_vertex(h);

        const surface_mesh::Point& p0 = m_mesh->position(v0);
        const surface_mesh::Point& p1 = m_mesh->position(v1);
        const surface_mesh::Point& p2 = m_mesh->position(v2);

        double area = 0.5 * std::abs((p1.head<2>()-p0.head<2>()).cross(p2.head<2>()-p0.head<2>()));
        
        double avg_sq_grad_norm = ( vtx_grads.row(v0.idx()).squaredNorm() + 
                                    vtx_grads.row(v1.idx()).squaredNorm() + 
                                    vtx_grads.row(v2.idx()).squaredNorm() ) / 3.0;

        cost[f.idx()] = (*m_input_density)[f.idx()] * area * avg_sq_grad_norm;
    }
}

void
TriangularMeshTransportSolver::print_debuginfo_iteration(int it, double alpha, double beta, ConstRefVector search_dir,
                               double l2err, ConstRefVector residual,
                               double t_linearsolve, double t_beta, double t_linesearch) const
{
  if(m_verbose_level>=6) {
    std::cout << "    execution time: search direction = " << t_linearsolve << "s,"
              << " beta = " << t_beta << "s,"
              << " line search = " << t_linesearch << "s,"
              << " [sum = " << t_linearsolve+t_beta+t_linesearch << "s]" << std::endl;
  }
  if(m_verbose_level>=5) {
    std::cout << "    alpha=" << alpha;
    std::cout << "  beta=" << beta;
    std::cout << "  norm(d)=" << search_dir.norm();
    std::cout << "  res^2=" << l2err << std::endl;
  }
  else if(m_verbose_level>=3)
  {
    std::cout << "    L2 residual^2 = " << l2err << std::endl;
  }
}

} // namespace otmap