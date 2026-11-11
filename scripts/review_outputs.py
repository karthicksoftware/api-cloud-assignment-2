"""Run automated semantic checks and prepare a readable member-review pack."""
import json
from retailassist.review import review_outputs

if __name__ == "__main__":
    print(json.dumps(review_outputs(), indent=2))
