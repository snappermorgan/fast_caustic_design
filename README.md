# Fast caustic design: A fast high contrast freeform optics designer based on OTMap

## Examples

#### Ring:
![ring simulation](data/ring_sim.png)

```
./caustic_design -res 512 -focal_l 1.5 -thickness 0.2 -width 1 -in_trg ../data/ring.png
```

#### Einstein:
![einstein simulation](data/einstein_sim.png)
*(Generated image)*

```
./caustic_design -res 512 -focal_l 1.5 -thickness 0.3 -width 1 -in_trg ../data/einstein.png
```

## Usage

Download the latest build from [Releases](https://github.com/dylanmsu/fast_caustic_design/releases). 

Run the command of one of the examples with the image locations properly filled out. After its complete the 3d model will be located in the directory above the directory of the exe.

### Progress Reporting

For integration with external tools or cloud runners, you can use the `--json-progress` flag.

```bash
./caustic_design -in_trg target.png --json-progress
```

When this flag is enabled, the application outputs structured JSON logs to `stdout` indicating the current status, stage, and progress percentage. This is useful for monitoring the job status remotely.

Example output:
```json
{"status": "processing", "stage": "surface_optimization", "progress": 0.35, "message": "Iteration 2/10"}
```

### Cloud Deployment

A Python wrapper (`cloud_runner.py`) is provided to facilitate cloud deployments (e.g., Google Cloud Platform). This script handles:

1.  Downloading input images from Google Cloud Storage (GCS).
2.  Executing the `caustic_design` binary with progress monitoring.
3.  Uploading the generated model back to GCS.

**Prerequisites:**

-   Python 3
-   Google Cloud SDK (if running locally) or a service account (in cloud).
-   Dependencies: `pip install -r requirements.txt`

**Usage:**

The wrapper accepts the same arguments as the C++ binary but supports `gs://` URIs for inputs and outputs.

```bash
python3 cloud_runner.py \
  --bin ./build/caustic_design \
  -in_trg gs://my-bucket/target.png \
  -output gs://my-bucket/results/my_lens.obj
```

This acts as a drop-in worker script for async queue runners that trigger backend processing with GCS paths.

### Configuration & Authentication

To execute `cloud_runner.py`, you must set up authentication for Google Cloud services (GCS and Cloud Logging).

**Environment Variables:**

1.  **`GOOGLE_APPLICATION_CREDENTIALS`**: Path to your service account JSON key file.
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
    ```
    *Ensure this service account has `Storage Object Admin` and `Logging Log Writer` roles.*

2.  (Optional) **`GCLOUD_PROJECT`**: If not inferred from credentials, set your project ID explicitely.

3.  (Optional) **`CLOUD_LOG_NAME`**: Set the custom log stream name for Cloud Logging (default: "python").
    ```bash
    export CLOUD_LOG_NAME="caustic-runner-worker"
    ```

**Cloud Logging:**
The runner automatically detects if `google-cloud-logging` is installed and attempts to send logs to Google Cloud Logging. This allows you to view progress and errors in the GCP Console under "Logs Explorer".

### Infrastructure as Code (Terraform)

The `terraform/` directory contains configuration to provision the complete Google Cloud environment.

**Resources Created:**
-   **Cloud Run Job**: Hosts the `caustic_design` container logs.
-   **Cloud Tasks**: Async queue to trigger jobs.
-   **Firestore**: Database for real-time progress tracking.
-   **Cloud Storage**: Buckets for inputs/outputs.
-   **IAM**: Service accounts with least-privilege access.

**Usage:**

1.  Initialize Terraform:
    ```bash
    cd terraform
    terraform init
    ```

2.  Plan (dry-run):
    ```bash
    terraform plan -var="project_id=YOUR_PROJECT_ID"
    ```

3.  Apply:
    ```bash
    terraform apply -var="project_id=YOUR_PROJECT_ID"
    ```

After applying, Terraform will output the created bucket names and resource IDs.
## Build from source

This code uses [Eigen](https://eigen.tuxfamily.org), Surface_mesh, and CImg that are already included in the repo/archive.
The only libraries you need to install are [Ceres Solver](http://ceres-solver.org/) for the normal integration and libpng/libjpg for image IO.

It is however highly recommended to install [SuiteSparse/Cholmod](http://faculty.cse.tamu.edu/davis/suitesparse.html) for higher performance.

### Install dependancies on windows:
Install vcpkg:
```bash
$ cd C:\
$ git clone https://github.com/microsoft/vcpkg.git
$ cd vcpkg
$ .\bootstrap-vcpkg.bat
```

Install Ceres solver and its dependancies:
```bash
$ ./vcpkg.exe install ceres[core] --triplet x64-windows-static
```

Install libpng
 and its dependancies:
```bash
$ vcpkg install libpng --triplet x64-windows-static
```

All you then need to do is to clone the repo, configure a build directory with cmake, and then build.

Make sure the path in CMAKE_TOOLCHAIN_FILE is correct.

````bash
$ git clone --recursive git@github.com:dylanmsu/fast_caustic_design.git
$ cd fast_caustic_design
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_BUILD_TYPE=Release
$ cmake --build . --config Release
$ ./Release/caustic_design.exe -res 512 -focal_l 1.5 -thickness 0.3 -width 1 -in_trg ../data/einstein.png
````

## How does it work
Creating a lens surface whose shadow matches a target image involves two main steps:

### 1. Figuring out how to move the light
First, we need to determine how light should travel from the lens surface to the target shadow. This is done using a technique called optimal transport (OT). OT is a challenging problem and an area of ongoing research. It gives us a map that tells each point on the lens surface where its light should land on the target image.

Traditionally, this problem is solved using structures called power diagrams (also known as Laguerre-Voronoi diagrams). However, in our case, we use a newer, faster method.

The usage of optimal transport is not just to reduce surface variations. It is actually quite important to have a truly optimal transport plan. According to Brenier's theorem, an optimal transport plan is the gradient of a potential. Without this fact we cannot approximate the inverse gradient (or normal integration) afterwards accurately.

### 2. Shaping the surface to steer the light
Once we have the transport map, the next step is to figure out the exact geometry of the lens surface. For each vertex on the surface, we use the OT map to set the x and y direction of the outgoing ray. The z-direction is chosen based on the focal length.

These rays define how we want the surface to bend the light. Using inverse Snell’s law, we compute the target surface normals needed to steer the rays correctly. Finally, we use a solver (Ceres) to adjust the vertex positions on the lens surface so that the computed normals match the target ones. This is called normal integration and can be thought of as approximating the inverse gradient of the vertex normals.
## Limitation
Currently, the code produces only square lenses.

The limitation stems from the fact that the OTMap solver is designed to compute the transport map from an image to a uniform distribution on the square domain (denoted as T(u->1)). This means we can only transport light from a square lens surface to an image.

To support arbitrary lens shapes like for example a circle, we need to tell the optimal transport solver to start from a circle and transport to an image. Here the circle is the source distribution (v), and the image is the target distribution (u).

The method that OTMap uses to compute a transport map from any source distribution to any target distribution is by a heuristic. They first move mass from a source distribution to a uniform distribution on the square domain (move mass along T(v->1)). Then transport the new mass from this uniform distribution on the square domain to the target distribution (moves new mass along T(1->u)). This is called composition, see equation 10 in the paper.

This approach is an estimation and does not yield a true optimal transport map. This estimation inadvertently introduces a small curl component into the mapping, so it is no longer purely the gradient of a potential.

Because deriving a heightmap for a lens relies on normal integration, which only utilizes the curl-free component of the mapping, the presence of any curl results in distortions in the caustic lens.

One solution to this issue would be to solve the transport map T(u->1) on a custom domain (think rounded rectangle, circle, ellipse, etc). This requires a rewrite because the current OTMap solver relies on a square domain with quad faces. You could use a triangular mesh as the domain and apply finite element analysis to compute the discrete differential operators. Namely the laplacian and the gradient. The laplacian uses a special stencil, and the gradient is calculated on the dual vertices, so this would not be trivial on a triangle mesh.

A second solution that may be more approachable is modifying the right hand side of equation 11 in the OTMap paper by replacing the cell integral of u(x) with the cell integral of u(x) / v(T(x)). This should solve the full Monge-Ampère equation and yield a true L2 optimal transport map T(u->v). The catch is that this makes the problem highly nonlinear and non-convex, thus slower or even no convergence in some cases.

Some other potential transport solving strategies that solve a single potential and thus guaranteed to not cause distortions are:
- A method that approximates the monge ampere operator by a monotone descretization. A wide stencil scheme for stable and monotone newton iterations.
- A method that reformulates the monge ampere equation to a scheme that is solved by fixed point updates of a potential. The potential is updated by the solution of a poisson equation. (FFT-OT)
- A method that alternates between two potential functions related by a convex dual transform (c-transform), performing gradient-based updates in each space to approximate an optimal transport mapping between two distributions. (BFM)

## License

The core of the transport solver is provided under the [GNU Public License v3](https://www.gnu.org/licenses/gpl-3.0.html).

Utilities and applications are released under the [Mozilla Public License 2](https://www.mozilla.org/en-US/MPL/2.0/).

## References

[1] Georges Nader and Gael Guennebaud. _Instant Transport Maps on 2D Grids_. ACM Transactions on Graphics (Proceedings of Siggraph Asia 2018). [[pdf]](https://hal.inria.fr/hal-01884157) [[video]](https://www.youtube.com/watch?v=Ofz4-reJQRk)
