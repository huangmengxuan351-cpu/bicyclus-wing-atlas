# Bicyclus Wing Atlas deployment source

This repository contains deployment code and reduced display resources for a Bicyclus anynana wing atlas. The approved release includes 12 genes, 291 image units, 72 atlas groups, 216 representations and six anatomical references. The hth dataset is excluded.

The deployment uses GitHub for source and CI, two independent Cloudflare Pages projects for Chinese and English browsing, and a ModelScope Docker Studio for Python prediction and password-protected original downloads. The repository contains no original scientific arrays, trained model bundles, full-resolution atlas images or real credentials.

**Both display sites are published; complete prediction acceptance is still pending.** Visit the [Chinese site](https://bicyclus-wing-atlas-zh.pages.dev/) or the [English site](https://bicyclus-wing-atlas-en.pages.dev/). Twenty-seven public HTTPS checks passed, including preview delivery and anonymous original-download denial. The private release contains 4,140 remote files whose sizes and SHA-256 hashes all match. ModelScope and Pages server-side Secrets are configured, and the free Docker backend is installing and verifying the full data release. GitHub source is published and its ModelScope Actions Secret is configured. Automated Pages publishing still needs the account-scoped Pages Edit token. No paid resource is selected automatically. See [the Chinese deployment guide](deploy/DEPLOYMENT_ZH.md) for acceptance boundaries and configuration.

## Layout

- `public/zh`, `public/en`: independent Pages sources, routing Worker and preview pack manifests.
- `code_python`: unchanged scientific prediction modules and interface sources.
- `output_results`: portable service wrappers and public interface files; no scientific data.
- `deploy/start.py`: persistent-data bootstrap and Linux service supervisor.
- `deploy/dispatcher.py`: the sole public API entry on `0.0.0.0:7860`.
- `deploy/upload_private_data.py`: verifies and uploads a release to an existing private dataset.
- `.github/workflows/pages.yml`: publishes both display projects.
- `.github/workflows/modelscope.yml`: updates an existing free Docker Studio.

Original data must remain in a **private** dataset and `/mnt/workspace/wing-atlas-data`; credentials and prediction histories remain in `/mnt/workspace/wing-atlas-state`. The download password belongs in a ModelScope **secret**, never in a public variable or committed `.env` file. Anonymous browsing and prediction remain available without a login. Original exports require the `atlas_download` account and its password.

The Python service keeps its private engines on loopback ports 8881, 8880, 8890 and 8891; only the dispatcher is exposed. The Studio root is an API service landing response. Visitors use the independent Pages interfaces. Preview images intentionally remain publicly viewable and saveable.

Programmatic access to the Docker Studio uses its `api-inference.modelscope.net` address. The Pages server adds its privately configured `MODELSCOPE_STUDIO_TOKEN`; visitors do not receive this credential or need a ModelScope login. Download authorization is forwarded separately and checked by the atlas backend. Initial setup reports `ready:false`; a running preparation container is not a successful prediction deployment.

## Data and scientific consistency

The private bundle includes a file-by-file SHA-256 manifest. Paths in deployment metadata are rebased to `/opt/wing-atlas`; model binaries and scientific arrays remain byte-identical to the approved local release. A software regression using an existing reference image does not measure predictive accuracy. The existing `needs_review` decision is preserved.

No license grant is implied for scientific data or model files by publication of this deployment source. The project owner should select the repository's software license before representing it as an open-source licensed release.
