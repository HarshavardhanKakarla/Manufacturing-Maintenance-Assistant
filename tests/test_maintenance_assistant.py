import unittest

from maintenance_assistant import (
    Document,
    DocumentChunk,
    chunk_documents,
    cosine_similarity,
    extract_documents,
    retrieve_chunks,
)


class DocumentPreparationTests(unittest.TestCase):
    def test_extracts_text_documents(self) -> None:
        docs = extract_documents([("manual.md", b"Keep guards in place.")])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].source, "manual.md")
        self.assertIsNone(docs[0].page)
        self.assertEqual(docs[0].text, "Keep guards in place.")

    def test_rejects_unsupported_file_types(self) -> None:
        with self.assertRaises(ValueError):
            extract_documents([("manual.docx", b"not supported")])

    def test_chunks_preserve_source_and_page(self) -> None:
        document = Document("manual.pdf", 3, "a" * 25)
        chunks = chunk_documents([document], chunk_size=10, overlap=2)
        self.assertEqual([len(chunk.text) for chunk in chunks], [10, 10, 9])
        self.assertTrue(all(chunk.source == "manual.pdf" for chunk in chunks))
        self.assertTrue(all(chunk.page == 3 for chunk in chunks))

    def test_rejects_invalid_chunk_overlap(self) -> None:
        with self.assertRaises(ValueError):
            chunk_documents([Document("manual.md", None, "text")], 10, 10)


class RetrievalTests(unittest.TestCase):
    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_retrieval_filters_weak_matches_and_sorts(self) -> None:
        chunks = [
            DocumentChunk("one.md", None, "motor ventilation"),
            DocumentChunk("two.md", 2, "belt tracking"),
        ]
        results = retrieve_chunks(
            chunks,
            [[1, 0], [0, 1]],
            [1, 0],
            minimum_similarity=0.5,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "one.md")

    def test_retrieval_rejects_mismatched_vectors(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_chunks(
                [DocumentChunk("one.md", None, "text")],
                [],
                [1, 0],
            )


if __name__ == "__main__":
    unittest.main()
