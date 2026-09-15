# Instance deployment — daily commands

The pipeline + checking UI are already set up on the GCP instance below.
This doc is just the commands you need for day-to-day use — no setup
steps here, that's already done.

- **Project**: `ozitech`
- **Instance**: `ozi-toys`
- **Zone**: `asia-south1-a`
- **Code on the instance**: `/home/mac/ozi_toys_image_pipeline`

## Start using it

1. Start the instance:
   ```bash
   gcloud compute instances start ozi-toys --zone=asia-south1-a --project=ozitech
   ```
   Wait ~30–60 seconds for it to boot.

2. Start the checking UI on the instance (auto-start on boot is
   intentionally OFF, so this manual step is required every time):
   ```bash
   gcloud compute ssh ozi-toys --zone=asia-south1-a --project=ozitech --command="sudo systemctl start ozi-streamlit.service"
   ```

3. Open a private tunnel from your Mac to the instance:
   ```bash
   gcloud compute ssh ozi-toys --zone=asia-south1-a --project=ozitech -- -L 8501:localhost:8501 -N
   ```
   This command does not print anything and does not return — that's
   normal, leave it running in its own terminal tab/window.

4. Open **http://localhost:8501** in your browser. That's the checking UI.

## Stop using it (do this when you're done for the day)

Stop the instance so it isn't billed while idle:
```bash
gcloud compute instances stop ozi-toys --zone=asia-south1-a --project=ozitech
```
This waits until it's fully stopped before returning. You can also just
close the tunnel terminal (Ctrl+C) any time — that only ends your access,
it doesn't stop billing, so still run the stop command above.

## Check status any time

```bash
gcloud compute instances describe ozi-toys --zone=asia-south1-a --project=ozitech --format="value(status)"
```
`RUNNING` = billing / usable once the tunnel is up. `TERMINATED` = fully
stopped, not billed for compute.

## If SSH or the tunnel suddenly stops working

- `Connection refused` on port 22 almost always means the instance is
  stopped or still shutting down — check status (above) before anything
  else.
- If you restart your Mac or close the tunnel terminal, `localhost:8501`
  stops working until you re-run the tunnel command in step 3 — the
  instance and the UI itself are unaffected, only your view of it.
- Starting the instance does NOT start the checking UI by itself
  (auto-start is off) — you must always run step 2 first, or
  `localhost:8501` will refuse to connect even though the instance is
  `RUNNING`.

## One-time setup (already done — for reference only)

- Code is deployed by copying the git-tracked files (`git ls-files | tar
  -czf ... | gcloud compute scp ...`) rather than cloning via GitHub auth
  on the instance.
- `.env`, `gcp_key.json`, `gcs_key.json` were copied separately via
  `gcloud compute scp` (never committed to git) and the paths inside
  `.env` were updated from the local Mac path to
  `/home/mac/ozi_toys_image_pipeline/...`.
- Streamlit runs as a systemd service (`ozi-streamlit.service`), bound to
  `127.0.0.1:8501` only — never exposed directly to the internet, only
  reachable through the SSH tunnel above. Auto-start on boot was
  deliberately turned off (`systemctl disable`), so it must be started
  manually each time (step 2 above) — start it with `systemctl start
  ozi-streamlit.service` and stop it early with `systemctl stop
  ozi-streamlit.service` if you ever need to free the port without
  stopping the whole instance.
