#!/usr/bin/env python3
"""
GitHub Project Finder

Searches GitHub for Java projects that are likely to compile and produce
Lower Bound Checker warnings. Uses GitHub API to discover repositories.
"""

import os
import json
import time
import logging
import argparse
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("Warning: PyGithub not available. Install with: pip install PyGithub")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GitHubProject:
    """Represents a GitHub project candidate"""
    name: str
    full_name: str
    url: str
    description: str
    stars: int
    language: str
    size: int  # Size in KB
    updated_at: str
    created_at: str
    license: Optional[str]
    topics: List[str]
    has_issues: bool
    has_wiki: bool
    default_branch: str
    clone_url: str
    ssh_url: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class GitHubProjectFinder:
    """Finds GitHub projects using GitHub API"""
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitHub project finder
        
        Args:
            github_token: GitHub personal access token (optional, increases rate limit)
        """
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.github = None
        
        if GITHUB_AVAILABLE:
            if self.github_token:
                self.github = Github(self.github_token)
                logger.info("Using GitHub API with authentication")
            else:
                self.github = Github()
                logger.info("Using GitHub API without authentication (limited rate)")
        else:
            logger.warning("PyGithub not available. Install with: pip install PyGithub")
    
    def search_repositories(self, 
                           query: str,
                           max_results: int = 100,
                           sort: str = 'stars',
                           order: str = 'desc') -> List[GitHubProject]:
        """
        Search GitHub repositories
        
        Args:
            query: GitHub search query (e.g., "language:java stars:>10")
            max_results: Maximum number of results to return
            sort: Sort field ('stars', 'updated', 'forks')
            order: Sort order ('desc', 'asc')
            
        Returns:
            List of GitHubProject objects
        """
        if not self.github:
            logger.error("GitHub API not available")
            return []
        
        projects = []
        
        try:
            logger.info(f"Searching GitHub with query: {query}")
            repositories = self.github.search_repositories(query, sort=sort, order=order)
            
            count = 0
            for repo in repositories:
                if count >= max_results:
                    break
                
                try:
                    # Get repository details
                    project = self._repo_to_project(repo)
                    projects.append(project)
                    count += 1
                    
                    # Rate limiting: GitHub allows 30 requests/minute for authenticated, 10 for unauthenticated
                    if count % 10 == 0:
                        time.sleep(1)  # Small delay to avoid hitting rate limits
                    
                except Exception as e:
                    logger.warning(f"Error processing repository {repo.full_name}: {e}")
                    continue
            
            logger.info(f"Found {len(projects)} repositories")
            
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
            if "rate limit" in str(e).lower():
                logger.error("GitHub API rate limit exceeded. Consider using a GitHub token.")
        
        return projects
    
    def _repo_to_project(self, repo) -> GitHubProject:
        """Convert GitHub repository object to GitHubProject"""
        return GitHubProject(
            name=repo.name,
            full_name=repo.full_name,
            url=repo.html_url,
            description=repo.description or "",
            stars=repo.stargazers_count,
            language=repo.language or "Unknown",
            size=repo.size,  # Size in KB
            updated_at=repo.updated_at.isoformat() if repo.updated_at else "",
            created_at=repo.created_at.isoformat() if repo.created_at else "",
            license=repo.license.name if repo.license else None,
            topics=list(repo.get_topics()),
            has_issues=repo.has_issues,
            has_wiki=repo.has_wiki,
            default_branch=repo.default_branch,
            clone_url=repo.clone_url,
            ssh_url=repo.ssh_url
        )
    
    def get_build_queries(self) -> List[str]:
        """Get list of search queries for finding Java projects"""
        queries = [
            # General Java projects with build files
            "language:java stars:>10 size:100..50000",
            "language:java pom.xml stars:>20",
            "language:java build.gradle stars:>15",
            
            # Projects likely to have array/index operations
            "language:java array index stars:>20",
            "language:java data structure stars:>15",
            "language:java algorithm stars:>10",
            "language:java collection stars:>10",
            "language:java util stars:>10",
            
            # Mathematical/numerical projects
            "language:java math stars:>10",
            "language:java numeric stars:>10",
            "language:java scientific stars:>10",
            
            # Recently updated projects (more likely to compile)
            "language:java pushed:>2023-01-01 stars:>10",
            "language:java updated:>2023-01-01 stars:>10",
        ]
        return queries
    
    def filter_projects(self, 
                       projects: List[GitHubProject],
                       min_stars: int = 10,
                       max_size_kb: int = 50000,
                       min_size_kb: int = 100,
                       updated_within_days: Optional[int] = None) -> List[GitHubProject]:
        """
        Filter projects by criteria
        
        Args:
            projects: List of projects to filter
            min_stars: Minimum number of stars
            max_size_kb: Maximum repository size in KB
            min_size_kb: Minimum repository size in KB
            updated_within_days: Only include projects updated within N days
            
        Returns:
            Filtered list of projects
        """
        filtered = []
        
        for project in projects:
            # Filter by stars
            if project.stars < min_stars:
                continue
            
            # Filter by size
            if project.size < min_size_kb or project.size > max_size_kb:
                continue
            
            # Filter by update date
            if updated_within_days:
                try:
                    updated_date = datetime.fromisoformat(project.updated_at.replace('Z', '+00:00'))
                    days_ago = (datetime.now(updated_date.tzinfo) - updated_date).days
                    if days_ago > updated_within_days:
                        continue
                except Exception:
                    pass  # Skip date filtering if parsing fails
            
            filtered.append(project)
        
        logger.info(f"Filtered {len(projects)} projects to {len(filtered)}")
        return filtered
    
    def save_projects(self, projects: List[GitHubProject], output_file: str):
        """Save projects to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        projects_dict = {
            'metadata': {
                'total_projects': len(projects),
                'generated_at': datetime.now().isoformat(),
                'source': 'github_api'
            },
            'projects': [project.to_dict() for project in projects]
        }
        
        with open(output_path, 'w') as f:
            json.dump(projects_dict, f, indent=2)
        
        logger.info(f"Saved {len(projects)} projects to {output_file}")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Find GitHub Java projects')
    parser.add_argument('--query', help='GitHub search query')
    parser.add_argument('--output', default='github_projects.json', help='Output JSON file')
    parser.add_argument('--max-results', type=int, default=100, help='Maximum results per query')
    parser.add_argument('--min-stars', type=int, default=10, help='Minimum stars')
    parser.add_argument('--max-size', type=int, default=50000, help='Maximum size in KB')
    parser.add_argument('--min-size', type=int, default=100, help='Minimum size in KB')
    parser.add_argument('--updated-within-days', type=int, help='Only projects updated within N days')
    parser.add_argument('--use-all-queries', action='store_true', help='Use all built-in queries')
    parser.add_argument('--github-token', help='GitHub personal access token')
    
    args = parser.parse_args()
    
    finder = GitHubProjectFinder(github_token=args.github_token)
    
    all_projects = []
    
    if args.use_all_queries:
        queries = finder.get_build_queries()
        logger.info(f"Using {len(queries)} built-in queries")
        
        for query in queries:
            projects = finder.search_repositories(query, max_results=args.max_results)
            all_projects.extend(projects)
            
            # Deduplicate by full_name
            seen = set()
            unique_projects = []
            for project in all_projects:
                if project.full_name not in seen:
                    seen.add(project.full_name)
                    unique_projects.append(project)
            all_projects = unique_projects
            
            # Rate limiting between queries
            time.sleep(2)
    elif args.query:
        projects = finder.search_repositories(args.query, max_results=args.max_results)
        all_projects = projects
    else:
        logger.error("Must provide --query or --use-all-queries")
        return 1
    
    # Filter projects
    filtered = finder.filter_projects(
        all_projects,
        min_stars=args.min_stars,
        max_size_kb=args.max_size,
        min_size_kb=args.min_size,
        updated_within_days=args.updated_within_days
    )
    
    # Save results
    finder.save_projects(filtered, args.output)
    
    logger.info(f"Found {len(filtered)} projects matching criteria")
    return 0

if __name__ == '__main__':
    exit(main())

