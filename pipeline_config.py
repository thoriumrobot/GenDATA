#!/usr/bin/env python3
"""
Pipeline Configuration
Default settings for the GenDATA pipeline with enhanced features.
"""

# Default slicer configuration
DEFAULT_SLICER_TYPE = 'soot'  # Use Soot slicer with forward/backward slicing
DEFAULT_SLICE_MODE = 'combined'  # Use combined forward+backward slicing

# Default augmentation configuration
DEFAULT_AUGMENTATION_TYPE = 'semantic'  # Use semantic-preserving transformations
DEFAULT_AUGMENT_FIRST = True  # Augment code before slicing
DEFAULT_AUGMENTATION_FACTOR = 10  # Number of variants to generate

# Default training configuration
DEFAULT_BASE_MODEL = 'enhanced_causal'
DEFAULT_EPISODES = 100
DEFAULT_DEVICE = 'auto'

# Default project paths
DEFAULT_PROJECT_ROOT = '/home/ubuntu/checker-framework/checker/tests/index'
DEFAULT_WARNINGS_FILE = '/home/ubuntu/GenDATA/index1.out'
DEFAULT_CFWR_ROOT = '/home/ubuntu/GenDATA'

# Slicer-specific configurations
SLICER_CONFIGS = {
    'soot': {
        'description': 'Enhanced Soot slicer with forward/backward slicing',
        'modes': ['backward', 'forward', 'combined'],
        'default_mode': 'combined',
        'features': [
            'Data flow analysis',
            'Control flow analysis', 
            'Def-use tracking',
            'Improved line mapping'
        ]
    },
    'specimin': {
        'description': 'Specimin slicer (legacy)',
        'modes': ['default'],
        'default_mode': 'default',
        'features': [
            'Basic slicing',
            'Legacy support'
        ]
    },
    'cf': {
        'description': 'Checker Framework slicer',
        'modes': ['default'],
        'default_mode': 'default', 
        'features': [
            'CFG-based slicing',
            'Checker Framework integration'
        ]
    }
}

# Augmentation configurations
AUGMENTATION_CONFIGS = {
    'semantic': {
        'description': 'Semantic-preserving transformations',
        'transformations': [
            'Loop conversions (for ↔ while)',
            'Guard reversals (if-else condition flipping)',
            'Mathematical properties (commutativity, identity operations)',
            'De Morgan\'s laws',
            'Ternary ↔ if-else conversions',
            'Switch ↔ if-else chain conversions',
            'Variable inlining/extraction'
        ],
        'preserves_semantics': True,
        'slicer_resistant': True
    },
    'random': {
        'description': 'Random code injection (legacy)',
        'transformations': [
            'Random code injection',
            'Synthetic variable generation',
            'Random method calls'
        ],
        'preserves_semantics': False,
        'slicer_resistant': False
    }
}

# Pipeline modes
PIPELINE_MODES = {
    'augment_first': {
        'description': 'Augment code first, then slice each variant',
        'flow': 'Original Code → Semantic Augmentation → Multiple Variants → Slicing → Slices',
        'benefits': [
            'Semantic diversity in slices',
            'Better slicer resistance',
            'More training data variety'
        ]
    },
    'traditional': {
        'description': 'Slice first, then augment slices',
        'flow': 'Original Code → Slicing → Slices → Augmentation → Augmented Slices',
        'benefits': [
            'Faster processing',
            'Legacy compatibility'
        ]
    }
}

def get_default_config():
    """Get the default pipeline configuration"""
    return {
        'slicer_type': DEFAULT_SLICER_TYPE,
        'slice_mode': DEFAULT_SLICE_MODE,
        'augmentation_type': DEFAULT_AUGMENTATION_TYPE,
        'augment_first': DEFAULT_AUGMENT_FIRST,
        'augmentation_factor': DEFAULT_AUGMENTATION_FACTOR,
        'base_model': DEFAULT_BASE_MODEL,
        'episodes': DEFAULT_EPISODES,
        'device': DEFAULT_DEVICE,
        'project_root': DEFAULT_PROJECT_ROOT,
        'warnings_file': DEFAULT_WARNINGS_FILE,
        'cfwr_root': DEFAULT_CFWR_ROOT
    }

def validate_config(config):
    """Validate a pipeline configuration"""
    errors = []
    
    # Validate slicer type
    if config.get('slicer_type') not in SLICER_CONFIGS:
        errors.append(f"Invalid slicer_type: {config.get('slicer_type')}")
    
    # Validate slice mode for the slicer
    slicer_type = config.get('slicer_type', DEFAULT_SLICER_TYPE)
    valid_modes = SLICER_CONFIGS[slicer_type]['modes']
    if config.get('slice_mode') not in valid_modes:
        errors.append(f"Invalid slice_mode '{config.get('slice_mode')}' for slicer '{slicer_type}'. Valid modes: {valid_modes}")
    
    # Validate augmentation type
    if config.get('augmentation_type') not in AUGMENTATION_CONFIGS:
        errors.append(f"Invalid augmentation_type: {config.get('augmentation_type')}")
    
    # Validate augmentation factor
    factor = config.get('augmentation_factor', DEFAULT_AUGMENTATION_FACTOR)
    if not isinstance(factor, int) or factor < 1:
        errors.append(f"Invalid augmentation_factor: {factor}. Must be a positive integer.")
    
    # Validate episodes
    episodes = config.get('episodes', DEFAULT_EPISODES)
    if not isinstance(episodes, int) or episodes < 1:
        errors.append(f"Invalid episodes: {episodes}. Must be a positive integer.")
    
    return errors

def print_config_summary():
    """Print a summary of the current configuration"""
    config = get_default_config()
    print("GenDATA Pipeline Configuration Summary")
    print("=" * 50)
    print(f"Slicer Type: {config['slicer_type']} ({SLICER_CONFIGS[config['slicer_type']]['description']})")
    print(f"Slice Mode: {config['slice_mode']}")
    print(f"Augmentation Type: {config['augmentation_type']} ({AUGMENTATION_CONFIGS[config['augmentation_type']]['description']})")
    print(f"Augment First: {config['augment_first']} ({PIPELINE_MODES['augment_first' if config['augment_first'] else 'traditional']['description']})")
    print(f"Augmentation Factor: {config['augmentation_factor']}")
    print(f"Base Model: {config['base_model']}")
    print(f"Episodes: {config['episodes']}")
    print(f"Device: {config['device']}")
    print()
    print("Key Features:")
    print(f"- {SLICER_CONFIGS[config['slicer_type']]['description']}")
    print(f"- {AUGMENTATION_CONFIGS[config['augmentation_type']]['description']}")
    print(f"- {PIPELINE_MODES['augment_first' if config['augment_first'] else 'traditional']['description']}")

if __name__ == "__main__":
    print_config_summary()
    
    # Validate configuration
    config = get_default_config()
    errors = validate_config(config)
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\n✓ Configuration is valid")
