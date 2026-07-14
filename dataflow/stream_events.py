import json
import argparse

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

SCHEMA = ("event_id:STRING,user_id:STRING,item_id:STRING,"
          "event_type:STRING,event_time:TIMESTAMP")


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)    # NOT --project (Beam owns that flag)
    ap.add_argument("--subscription", required=True)
    args, beam_args = ap.parse_known_args()

    opts = PipelineOptions(
        beam_args,
        streaming=True,
        save_main_session=True,    # ships the top-level `json` import to the workers
        project=args.project_id,
    )

    with beam.Pipeline(options=opts) as p:
        (p
         | "Read" >> beam.io.ReadFromPubSub(subscription=args.subscription)
         | "Parse" >> beam.Map(lambda m: json.loads(m.decode("utf-8")))
         | "ToBQ" >> beam.io.WriteToBigQuery(
             f"{args.project_id}:reco.events",
             schema=SCHEMA,
             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
             create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER))


if __name__ == "__main__":
    run()