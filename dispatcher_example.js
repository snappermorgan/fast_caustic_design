const { Firestore } = require('@google-cloud/firestore');
const { CloudTasksClient } = require('@google-cloud/tasks');

const firestore = new Firestore();
const tasksClient = new CloudTasksClient();

const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT;
const QUEUE_LOCATION = 'us-central1';
const QUEUE_NAME = 'caustic-design-queue';
const CLOUD_RUN_JOB_URL = 'https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/' + PROJECT_ID + '/jobs/caustic-runner:run';
const JOB_REGION = 'us-central1';
// Note: Cloud Run Jobs are triggered via the v2 API usually, or via run.googleapis.com
// A simple way is to use an HTTP target task that calls a small Cloud Function or standard endpoint,
// BUT for Cloud Run Jobs specifically, the proper way is creating a task with http_request to the Google API
// OR arguably simpler: The task payload *is* the config, and the worker is a Cloud Run Service that runs the extensive job.
// HOWEVER, the Architecture Plan specified Cloud Run Jobs.
// Cloud Run Jobs API trigger via Cloud Tasks is complex because it requires an OAuth token with IAM permissions.
//
// Below is a simplified example assuming we have a Cloud Function or Service that proxies the creation of the Execution
// OR we use the Cloud Run Jobs REST API directly.

async function dispatchJob(userId, imageGcsUri, config) {
    // 1. Create Firestore Record
    const jobRef = firestore.collection('jobs').doc();
    await jobRef.set({
        userId: userId,
        status: 'QUEUED',
        createdAt: Firestore.Timestamp.now(),
        config: {
            input_uri: imageGcsUri,
            ...config
        }
    });

    console.log(`Job created: ${jobRef.id}`);

    // 2. Construct the Job Execution Request
    // We will call the Cloud Run Jobs "run" method via the REST API.

    // The command arguments to pass to the container
    const containerArgs = [
        '--bin', '/app/apps/build/caustic_design',
        '-in_trg', imageGcsUri,
        '-output', `gs://my-bucket/results/${jobRef.id}_output.obj`,
        '--firestore-doc', `jobs/${jobRef.id}`
    ];

    if (config.resolution) containerArgs.push('-res', config.resolution.toString());
    if (config.focal_l) containerArgs.push('-focal_l', config.focal_l.toString());
    // ... other args

    // 3. Create Task
    const parent = tasksClient.queuePath(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME);

    // We target the Cloud Run Jobs API directly.
    // https://cloud.google.com/run/docs/reference/rest/v1/namespaces.jobs/run
    const url = `https://${JOB_REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/caustic-runner:run`;

    const task = {
        httpRequest: {
            httpMethod: 'POST',
            url: url,
            oauthToken: {
                serviceAccountEmail: process.env.INVOKER_SERVICE_ACCOUNT
            },
            headers: {
                'Content-Type': 'application/json',
            },
            // Override the container args in the job execution
            body: Buffer.from(JSON.stringify({
                overrides: {
                    containerOverrides: [
                        {
                            args: containerArgs
                        }
                    ]
                }
            })).toString('base64')
        }
    };

    const [response] = await tasksClient.createTask({ parent, task });
    console.log(`Created queue task: ${response.name}`);

    return jobRef.id;
}

// Example Usage
// dispatchJob('user_123', 'gs://bucket/input.png', { resolution: 512, focal_l: 1.5 });
