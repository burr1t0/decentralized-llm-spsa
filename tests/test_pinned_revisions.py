import re
import unittest

from src import dataset_loader, it_dataset_loader, prompt_embeddings

SHA = re.compile(r"^[0-9a-f]{40}$")


class TestPinnedRevisions(unittest.TestCase):
    def test_corpus_revisions_are_commit_hashes(self):
        for dataset, revision in dataset_loader.HF_REVISIONS.items():
            self.assertRegex(revision, SHA, dataset)

    def test_it_sources_are_pinned(self):
        for skill, source in it_dataset_loader.DATASET_SOURCES.items():
            pinned = source.revision or (source.data_file and re.search(r"@([0-9a-f]{40})/", source.data_file).group(1))
            self.assertRegex(pinned or "", SHA, skill)

    def test_encoder_is_pinned(self):
        self.assertRegex(prompt_embeddings.ENCODER_REVISION, SHA)


if __name__ == "__main__":
    unittest.main()
