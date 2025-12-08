#!/usr/bin/env python3
"""
SQL Quotes Checker Implementation

This module implements the CheckerInterface for the SQL Quotes Checker,
which tracks quote parity in SQL query strings.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import logging
from checker_interface import CheckerInterface

logger = logging.getLogger(__name__)

class SqlQuotesChecker(CheckerInterface):
    """Implementation of CheckerInterface for SQL Quotes Checker"""
    
    def get_checker_name(self) -> str:
        return "SqlQuotes"
    
    def get_checker_processor(self) -> str:
        return "org.checkerframework.checker.quotes.QuotesChecker"
    
    def get_annotation_types(self) -> List[str]:
        return ['@SqlEvenQuotes', '@SqlOddQuotes']
    
    def parse_warnings(self, warnings_file: str) -> List[Dict[str, Any]]:
        """
        Parse SQL Quotes Checker warnings from output file.
        
        Warning format example:
        SqlQuery.java:23: error: [quotes.unsafe] SQL query string has odd number of quotes
        """
        warnings = []
        
        if not Path(warnings_file).exists():
            logger.warning(f"Warnings file not found: {warnings_file}")
            return warnings
        
        try:
            with open(warnings_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse warning line
                    match = re.match(r'(.+?):(\d+):\s*(error|warning):\s*\[(.+?)\]\s*(.+)', line)
                    if match:
                        file_path, line_num, level, checker_msg, message = match.groups()
                        warnings.append({
                            'file': file_path,
                            'line': int(line_num),
                            'column': 0,
                            'level': level,
                            'checker_message': checker_msg,
                            'message': message,
                            'annotation_type': self._infer_annotation_type(checker_msg, message)
                        })
        except Exception as e:
            logger.error(f"Error parsing warnings file {warnings_file}: {e}")
        
        return warnings
    
    def _infer_annotation_type(self, checker_msg: str, message: str) -> Optional[str]:
        """Infer likely annotation type from warning message"""
        msg_lower = message.lower()
        
        if 'odd' in msg_lower or 'unsafe' in msg_lower:
            return '@SqlOddQuotes'
        elif 'even' in msg_lower:
            return '@SqlEvenQuotes'
        
        return None
    
    def extract_features(self, cfg_data: Dict[str, Any], node: Dict[str, Any]) -> List[float]:
        """
        Extract SQL Quotes Checker-specific features.
        
        Features include:
        - Quote count in string literals
        - String concatenation patterns
        - SQL method call patterns
        - Sanitization method calls
        """
        features = []
        label = node.get('label', '').lower()
        node_type = node.get('node_type', '').lower()
        
        # Feature 1: String literal with quotes
        has_quotes = "'" in label or '"' in label
        features.append(1.0 if has_quotes else 0.0)
        
        # Feature 2: Quote count (even/odd)
        single_quote_count = label.count("'")
        double_quote_count = label.count('"')
        total_quotes = single_quote_count + double_quote_count
        is_even_quotes = (total_quotes % 2 == 0) if total_quotes > 0 else True
        features.append(1.0 if is_even_quotes else 0.0)
        features.append(float(total_quotes))  # Quote count
        
        # Feature 3: String concatenation
        is_concatenation = '+' in label and ('string' in node_type or 'str' in label)
        features.append(1.0 if is_concatenation else 0.0)
        
        # Feature 4: SQL method call
        is_sql_method = any(pattern in label for pattern in [
            'executequery', 'executeprepared', 'executeupdate', 'preparedstatement',
            'statement.execute', 'connection.prepare'
        ])
        features.append(1.0 if is_sql_method else 0.0)
        
        # Feature 5: Sanitization method
        is_sanitization = any(pattern in label for pattern in [
            'quote(', 'escape(', 'sanitize(', 'escapeSql'
        ])
        features.append(1.0 if is_sanitization else 0.0)
        
        # Feature 6: Prepared statement pattern
        is_prepared = 'preparedstatement' in label or 'preparestatement' in label
        features.append(1.0 if is_prepared else 0.0)
        
        return features
    
    def validate_annotation(self, annotation_type: str, location: Dict[str, Any]) -> bool:
        """Validate annotation placement"""
        if annotation_type not in self.get_annotation_types():
            return False
        
        target_type = location.get('target_type', '')
        valid_targets = ['parameter', 'local_variable', 'return']
        
        return target_type in valid_targets
    
    def get_training_data_source(self) -> str:
        # SQL Quotes Checker test suite location
        return '/home/ubuntu/checker-framework/checker/tests/quotes/'
    
    def get_warning_patterns(self) -> List[str]:
        return [
            'quotes.unsafe',
            'quotes.odd',
            'quotes.even',
            'sql.unsafe',
            'quotes'
        ]

