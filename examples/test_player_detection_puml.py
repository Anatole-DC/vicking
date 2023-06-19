from river.lib.utils.plantuml_export import export_plantuml

from examples.player_detection import PlayerDetectionPipeline

if __name__ == "__main__":
    export_plantuml(PlayerDetectionPipeline)