from src.pipeline import RagPipeline


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def answer_with_retrieval(self, question, context_chunks):
        self.calls.append((question, context_chunks))
        return "Grounded answer"


class FakeRetriever:
    def retrieve(self, question, k):
        return ["Relevant passage"], [0.25]


def test_pipeline_generates_only_grounded_answer():
    pipeline = RagPipeline()
    generator = FakeGenerator()
    pipeline._get_generator = lambda: generator
    pipeline._get_retriever = lambda name: FakeRetriever()

    result = pipeline.ask("Question", k=1)

    assert result["answer_with_retrieval"] == "Grounded answer"
    assert generator.calls == [("Question", ["Relevant passage"])]
    assert "answer_without_retrieval" not in result
    assert set(result["timing_seconds"]) == {"retrieval", "generation"}