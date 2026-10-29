"""Contrôles de cohérence du dépôt, du déploiement et de l'interface."""
import re

import pytest


def read(project_root, relative_path):
    return (project_root / relative_path).read_text(encoding="utf-8")


def test_version_file_is_semantic(project_root):
    assert re.fullmatch(r"\d+\.\d+\.\d+", read(project_root, "VERSION").strip())


def test_runtime_dependencies_exclude_heavy_unused_packages(project_root):
    packages = [
        line.split("#", 1)[0].strip().lower()
        for line in read(project_root, "requirements.txt").splitlines()
    ]
    packages = [line for line in packages if line]
    assert not any(line.startswith(("sentence-transformers", "torch")) for line in packages)


def test_docker_image_only_copies_runtime_files(project_root):
    dockerfile = read(project_root, "Dockerfile")
    for needed in ("COPY src", "COPY static", "COPY Data", "VERSION", "docker-entrypoint.sh"):
        assert needed in dockerfile
    dockerignore = read(project_root, ".dockerignore")
    for excluded in (".git", "docs", "experiments", "tests"):
        assert excluded in dockerignore.split()


def test_space_metadata_matches_container_port(project_root):
    readme = read(project_root, "README.md")
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    assert "7860" in read(project_root, "docker-entrypoint.sh")


def test_deployment_waits_for_tests(project_root):
    workflow = read(project_root, ".github/workflows/sync-to-hub.yml")
    assert "needs: tests" in workflow
    assert "refs/heads/main" in workflow


def test_no_duplicate_python_modules_at_repository_root(project_root):
    assert not list(project_root.glob("*.py"))


def test_frontend_explains_scores_and_citation_audit(project_root):
    html = read(project_root, "static/index.html")
    for expected in (
        "available_but_uncited_source_ids",
        "source(s) disponible(s) mais non citée(s)",
        "Similarité cosinus TF-IDF",
        "pas une probabilité",
        "hybrid_rank_score",
        "Score de fusion des rangs",
        "pipeline_metrics",
        "Durées observées (ms)",
        'href="/claims"',
    ):
        assert expected in html
    assert "reranker" not in html.lower()


def test_claims_dashboard_marks_data_as_synthetic(project_root):
    html = read(project_root, "static/claims.html")
    assert "tous les dossiers affichés sont fictifs" in html
    assert "décision" in html.lower()


@pytest.mark.parametrize("document", ["README.md", "docs/guide_examinateur.md"])
def test_examiner_documentation_lists_checks(project_root, document):
    text = read(project_root, document)
    for endpoint in ("/health", "/ready", "/version", "/search", "/claims"):
        assert endpoint in text
