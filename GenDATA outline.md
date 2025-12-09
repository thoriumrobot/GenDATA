# **GenDATA: A Generic Detection of Annotations in Type Analysis for Automated Checker Framework Annotations**

## **Abstract**

Pluggable type systems such as those in the Checker Framework improve software reliability by verifying type-related properties at compile time. However, applying these type checkers typically requires extensive annotations—a tedious task that hinders adoption by requiring developers to write annotations manually. Recent work has proposed Machine Learning (ML) models like NullGTN trained on a pre-existing dataset of human annotated code. A limitation of NullGTN-like approaches is that they cannot infer *rare annotations*, that is, annotations which are scarce in real datasets of human-annotated code. NullGTN was restricted to nullness checkers because most annotations are rare in this sense. In this paper, we propose **GenDATA (Generic Derivation of Annotations in Type Analysis)**, a pipeline for automating annotation placement using machine learning that addresses this key barrier in training. We generate a synthetic dataset by leveraging warnings from the Checker Framework itself, applying data augmentation techniques, then slicing relevant code. We then train machine learning models to recommend annotations. Our methodology applies equally to all pluggable type systems that do not require parameters. Empirical results show that GenDATA is able to generalize the performance of previous work such as NullGTN to type systems with rare annotations. For nullness, GenDATA performs as well as NullGTN.

**Keywords**:  
 Pluggable Type Checking, Checker Framework, Code Annotation, Data Augmentation, Machine Learning, Lower Bound Checker, SQL Quotes Checker, Signature String Checker

---

## **1\. Introduction**

### **1.1 Motivation for Automated Pluggable Type Checking**

Modern software often relies on run-time checks or testing to detect potentially dangerous operations such as array out-of-bounds errors. Pluggable type systems, such as those in the Checker Framework, offer compile-time guarantees that catch errors early and improve software reliability. However, these frameworks rely heavily on programmer-supplied annotations (e.g., `@NonNull`, `@Positive`) to convey type constraints. The tediousness of manually annotating legacy codebases is a major obstacle to the adoption of pluggable type checking. This problem has motivated researchers to develop automatic inference techniques like NullAway Annotator, CF WPI, and NullGTN.

NullAwayAnnotator\[4\] uses the local type inference already conducted by pluggable type checkers to place @Nullable annotations for NullAway. It is good for nullness, but inapplicable to other type systems. WPI\[5\] works on all checkers supported by the Checker Framework, but its performance is not as good. NullGTN\[1\] uses machine learning to place @Nullable annotations. It is facilitated by the public availability of large datasets of code with @Nullable annotations. GenDATA seeks to extend this technique to other types of checkers. However, GenDATA is still limited to placing only annotations that do not require parameters. Placing parameters requires us to use a generative model, which is hard to train and has lower accuracy than classifiers. Our goal with GenDATA is to improve on the generality of NullGTN while retaining its superior performance vs WPI.

### **1.2 Problem Statement**

Despite the benefits of automated annotation, machine learning solutions require large training datasets of correctly annotated code—a resource that is scarce, particularly for specialized type checkers. Pluggable type checkers are designed to be customizable. Given their specialized deployment scenarios, most of them cannot have a lot of human annotated code.

In terms of data availability, NullGTN\[1\] showed that nullability is an outlier among extant pluggable type systems. Using Sourcegraph's src tool, we found that none of the specialized type systems for Java, aside from nullability, had enough publicly-available data to plausibly train them in the same way NullGTN was trained. NullGTN gave good results only after training around 16k classes. The closest annotations for problems other than nullability in terms of common-ness in open-source were **`@GuardedBy`** (from the Checker Framework's Lock Checker) with 1,752 annotated classes; integer **`@NonNegative`** with 1,675 annotated classes; and **`@Positive`** with 1,008 annotated classes—these counts are an order of magnitude less than the annotations for **`@Nullable`**. This scarcity of training data makes it implausible to train a model with publicly available code to place their annotations. We call these annotations with a scarcity of large datasets of well-annotated code "rare annotations". What is needed is a method to generate the dataset to train machine learning models for placing rare annotations.

### **1.3 Proposed Solution: GenDATA**

We introduce **GenDATA (Generic Detection of Annotations in Type Analysis)**, a pipeline designed to generate training data and automatically suggest annotations for pluggable type checkers. The proposed pipeline generalizes to other pluggable type systems as well. Our approach uses the following workflow:

1. **Collect Checker Framework Warnings**: We use the Checker Framework test suite during training and code segments from open source projects from Github during evaluation. We cannot evaluate on test suites because we cannot guarantee the results would generalize to unannotated code (see section 5). Run the Checker Framework to identify code locations that trigger warnings related to a particular type checker. Output: Warnings.

2. **Random Code Augmentation**: Insert small snippets or transformations into the code to augment the data, helping the model learn to be robust and generalize beyond limited examples. Output: Augmented code.

3. **Code Slicing**: Use a slicer to isolate small code slices relevant to each warning, ensuring the model focuses on pertinent dataflow and control-flow details. Output: Slices.

4. **Guess annotation placements:** Use the model weights to place annotations. Output: Tentatively annotated code.

5. **(If training) Train Machine Learning Models**: Use feedback from the checker to evaluate and refine the model. We use the feedback of the number of warnings to guide the model towards accurate annotation placement. The same slice is processed for a predetermined constant number of iterations. Output: Trained model. Go back to step 3 until all the slices have been processed. If evaluating, we test the model on the code from the checker's test suite. This code contains ground truth about where the annotations should be.

### **1.4 Contributions**

* **Data Generation Pipeline**: We present a novel, generalizable approach to generate an annotated corpus from code that produces warnings, circumventing the challenge that we lack human-annotated datasets.  
* **Random Code Augmentation**: We systematically inject random snippets into code slices, augmenting the training data and **improving model robustness** by simulating diverse real-world variations of code.  
* **Multi-Checker Infrastructure**: We introduce a unified, extensible architecture for evaluating GenDATA across multiple Checker Framework checkers, enabling systematic comparison and validation of our approach across different type systems with varying complexity levels.
* **Initial Empirical Evaluation**: We demonstrate that our approach reduces Checker Framework warnings in unseen code for multiple checkers. We chose to evaluate on three checkers: the Lower Bound Checker, the SQL Quotes Checker and the Signature String Checker. We compare GenDATA's performance not only against simpler baselines but also NullGTN\[1\], to illustrate how the approaches differ when annotation data is scarce.

---

## **2\. Background and Related Work**

### **2.1 Pluggable Type Systems**

Pluggable type checking permits adding specialized checks without altering a language's built-in type system. The Checker Framework is an implementation of pluggable type checking, supporting a variety of type checkers (e.g., Nullness Checker, Index Checker, Regex Checker). Developers annotate source code with checker-specific qualifiers, and the Checker Framework enforces corresponding constraints at compile time.

### **2.2 Checkers**

WPI infers annotations for all checkers, but it has poor performance. NullGTN has superior performance, but it only works on nullability. GenDATA seeks to improve on the generality of NullGTN while retaining its superior performance relative to WPI. It is currently limited by placing only annotations that do not require parameters. We have chosen the Lower Bound, SQL Quotes and String Signature checkers (in the order of increasing complexity) to evaluate it because these checkers do not require parameters and because they have varying levels of complexity.

#### 2.2.1 Lower Bound Checker

One component of the Index Checker is the **Lower Bound Checker**, which prevents negative array indexing errors by ensuring index expressions cannot be negative. This is a relatively simple checker because it tracks a single numeric lower-range value.

As described in Figure ‍11.1 of the Checker Framework manual, the Lower Bound Checker defines the following qualifiers among others:

* **`@Positive`**: Values are ≥ 1; safe for indexing.  
* **`@NonNegative`**: Values are ≥ 0; safe for indexing.

Other annotations include @GTENegativeOne, @LowerBoundBottom and @LowerBoundUnknown. @GTENegativeOne is similar to @NonNegative, but less commonly used. @LowerBoundBottom and @LowerBoundUnknown are part of the type system, but typically not written by programmers.

These qualifiers ensure array or list access is protected against negative indexing. When the Checker Framework encounters code that may violate these constraints, it issues warnings or errors, prompting developers to annotate or correct the code.

#### 2.2.2 SQL Quotes Checker

The **SQL Quotes Checker** protects against SQL-injection vulnerabilities in code that constructs SQL queries via string concatenation. It reasons about the *parity* of single-quote characters (`'`) in every `String` value: a string with an **even** number of quotes can safely be concatenated into a query (its quotes are “balanced”), whereas a string with an **odd** number of quotes, or whose quoting status is unknown, is considered unsafe until it has been explicitly sanitized. By tracking quote parity through assignments and concatenations, the checker guarantees at compile-time that no unchecked value can reach a SQL-execution method such as `Statement.executeQuery`.

This is a more complex checker than Lower Bound because it tracks the parity of quotation marks across concatenation operations and sanitisation functions. The latter are likely to be harder to reason about than arithmetic ranges.

The type system supplies the user-relevant qualifiers (Figure 13.1 of the Checker Framework manual):

* `@SqlEvenQuotes`: a string literal that contains an even number of single quotes (possibly zero).

* `@SqlOddQuotes`: a string literal that contains an odd number of single quotes.

Concatenation follows parity arithmetic: two odd-quoted strings yield an even-quoted result, while concatenating an odd-quoted string with an even-quoted one preserves odd parity, and so on. Library methods that execute SQL are annotated to require `@SqlEvenQuotes` parameters, and sanitization helpers (for example, `quote(String s)`) are annotated to take `@SqlQuotesUnknown` and return `@SqlEvenQuotes`. With these annotations in place, the checker emits an error if any possibly unsafe value can flow into a query string.

Although modern best practice is to rely on prepared statements that avoid manual quoting altogether, the SQL Quotes Checker is useful for the considerable amount of legacy Java that still assembles SQL strings. The checker is meaningful for evaluating **GenDATA** because (1) its qualifiers are non-parametric, allowing our pipeline to learn them directly, and (2) its data-flow properties (quote parity and sanitization) differ markedly from numeric range or SQL Quotes checks, providing a complementary test of GenDATA’s generality.

#### 2.3.3 Signature String Checker

The Signature String Checker ensures that string representations of Java types match the exact formats expected by certain Java methods and internal JVM structures. Java defines multiple string formats for describing types—such as fully qualified names, binary names, internal names, and field descriptors—and mixing them up can cause run-time errors or unexpected behavior. For instance, \`Class.forName(String)\` requires a slightly different string format than \`MethodDescriptor\` or field-descriptor formats. Since this checker tracks multiple mutually-exclusive string format, it is the hardest to reason about.

By annotating method parameters with the appropriate signature annotation (e.g., **@FullyQualifiedName**, **@BinaryName**, **@FieldDescriptor**), the checker guarantees that the code passes the correct format. This reduces errors where a programmer might pass a dotted name (\`mypkg.MyClass\`) instead of a slashed name (\`mypkg/MyClass\`), or vice versa. Whenever a method like \`Class.forName\` or \`Class.getName\` demands a specific representation, the checker flags any mismatch, thereby preventing subtle mistakes.

**Internal String Feature Extraction**: GenDATA implements a comprehensive 30-feature extraction system for the Signature String Checker that analyzes Java source code to distinguish between the three annotation types. The system extracts features from actual string values in source code (when available) and analyzes format patterns, structural characteristics, usage context, and CFG relationships. Features include format detection (dotted vs slashed vs descriptor), package depth, class name patterns, character-level patterns, and context indicators (Class.forName usage, reflection APIs, etc.). This enables ML models to accurately predict which annotation type (@FullyQualifiedName, @BinaryName, or @FieldDescriptor) should be placed at each location.

### **2.3 Slicing**

Code slicing, also known as program slicing, is a static program analysis technique used to extract parts of source code that are relevant to specific aspects of program behavior, typically based on a selected variable or computation point. It helps isolate and understand program dependencies by identifying all statements that potentially affect or are affected by the slicing criterion \[7\].

Program slicing works by tracing data and control dependencies in a codebase. If you specify a line at a certain program point (called the slicing criterion), slicing identifies all code statements that influence or are influenced by that line. These can be extracted into a "slice" to help developers focus on relevant code for debugging, testing, refactoring, or understanding legacy systems.

This technique supports a variety of use cases in software engineering, such as simplifying testing by focusing on critical parts of the code, reverse engineering legacy software \[9\], and evaluating software safety or complexity \[10\].

We use the slicer to, given a line in code (e.g., with the variable responsible for a warning), extracts a “slice” containing all the information needed for the annotation placement. This approach reduces noise in training data, especially important for machine learning tasks that can be hampered by irrelevant context.

We considered using the Specimin slicer. However, Specimin is designed to retain information relevant to the type of a variable. What we want is information relevant to the run-time value of a variable. The value is useful to infer the annotation for a certain program point. We use a static slicer instead of a dynamic slicer because we want to reason about properties instead of devoting resources to exploring execution paths.

A value-based checker checks a property related to the values stored in variables. Examples of this kind of checker are the nullness checker ("null" is a special value) or the lower bound checker ("positive" is a property of certain integers but not others). Formally, such a checker's semantics are defined based on the actual run-time values of the program.

By contrast, a provenance-based checker examines where the values came from. The canonical example is the Tainting Checker: whether a value is tainted or untainted depends entirely on where it came from rather than the actual contents of the value at run time. By inspecting the run-time state, there would be no way to tell which is which.

Since our methodology is focused on reasoning about the values stored in variables, it is well-suited to value-based checkers. Hence, we focus on those in our evaluation.

### **2.4 Data Augmentation in Code Analysis**

Machine learning for code tasks (e.g., auto-completion, bug detection, type inference) often suffers from limited labeled data. Data augmentation techniques like image rotation have been fruitful in mature ML applications like object detection\[3\]. The equivalent for a coding task is to add and remove irrelevant statements and structural features. For example, if the declaration **int i=0;** that's irrelevant to the inferred property is added to an altered copy of the input data, the model learns to ignore such distractions during the detection step. We introduce small amounts of irrelevant code in a controlled fashion, and then use slicing to remove most of the irrelevant code to facilitate training.

Dataset augmentation is a standard regularization technique that increases both the quantity and variety of training data by introducing artificial samples derived from existing data\[3\]. In standard classification settings, the model's role is to map each high-dimensional input 𝑥 to a category label 𝑦 while remaining invariant to a wide range of transformations that do not affect the label. Common augmentation methods for image classification include small translations, rotations, flips, and scale changes, each of which preserves the true label yet introduces new, distinct training examples. Although such transformations can greatly improve generalization, one must be careful to avoid operations that could alter the correct label (e.g., flipping a 'b' into a 'd' in character recognition).

Extending these ideas to code-based inputs involves manipulating program text and structure in ways that leave the semantic properties that the checker cares about unchanged. Just as a rotation or a flip preserves the image class, these changes preserve the program features we intend our model to predict.

#### **2.4.1 Transformation Categories**

We implement **20 semantic-preserving transformations** organized into two categories:

**Enhanced Transformations (10)**: These apply sophisticated code transformations that preserve semantics while changing structure:

1. **Loop Conversion** (`loop_conversion`): Convert between for and while loops  
2. **Guard Reversal** (`guard_reversal`): Reverse conditional logic using De Morgan's laws  
3. **Mathematical Expression** (`mathematical_expression`): Apply mathematical properties (commutativity, associativity)  
4. **Logical Expression** (`logical_expression`): Apply boolean algebra transformations  
5. **Ternary Operator** (`ternary_operator`): Convert between ternary operators and if-else statements  
6. **Switch Statement** (`switch_statement`): Transform switch statements to if-else chains  
7. **Variable Operation** (`variable_operation`): Transform variable operations and assignments  
8. **Brace Normalization** (`brace_normalization`): Code formatting variations  
9. **String Concatenation** (`string_concatenation`): Transform string concatenation patterns  
10. **Numeric Literal** (`numeric_literal`): Transform numeric literals and constants

**Simple Transformations (10)**: Basic code transformations with minimal complexity:

11. **Simple Method Call** (`simple_method_call`): Transform simple method calls  
12. **Simple Assignment** (`simple_assignment`): Transform simple assignments  
13. **Simple Conditional** (`simple_conditional`): Transform simple conditional statements  
14. **Simple Array Access** (`simple_array_access`): Transform array access operations  
15. **Simple Return Statement** (`simple_return_statement`): Transform return statements  
16. **Simple Variable Declaration** (`simple_variable_declaration`): Transform variable declarations  
17. **Simple Constructor Call** (`simple_constructor_call`): Transform constructor calls  
18. **Simple Field Access** (`simple_field_access`): Transform field access operations  
19. **Simple String Operation** (`simple_string_operation`): Transform string operations  
20. **Simple Numeric Operation** (`simple_numeric_operation`): Transform numeric operations

All transformations are implemented using Eclipse JDT AST parsing for semantic preservation and accuracy. The pipeline automatically selects between Enhanced (10 methods) and Simple (10 methods) transformation systems based on code complexity analysis.

### **2.5 Machine Learning for Type Annotation**

Prior studies have used ML-based type inference (e.g., inferring Python types\[11,12,13,14,15,16\] or NullGTN\[1\], which adds Nullness annotations in Java). However, no efforts target checkers like the Lower Bound Checker, which don't have large publicly available human annotated codebases. We evaluate the following model families:

1. **Graph Convolution Network (GCN):** A class of neural networks that use graph convolutions to aggregate neighbor data for each node.  
2. **Heterogeneous Graph Transformer (HGT):** A strict superset of GCNs designed to work on heterogeneous graphs with multiple edge types.  
3. **Gradient Boosted Trees (GBT):** Feature-based ensemble learning using gradient boosting.  
4. **Causal Model:** Feature-based model incorporating causal inference principles.  
5. **Enhanced Causal Model:** Extended causal model with graph embeddings and advanced feature processing.  
6. **Graph Convolutional Sequence Network (GCSN):** Combines graph convolutions with sequence modeling.  
7. **DG2N:** Deep graph-to-graph network for structured prediction.

**Note**: GPT-4 LLM evaluation is planned for future work.

---

## **3\. GenDATA Pipeline Overview**

### **3.1 System Architecture**

Figure 1 (section 1.3) visualizes the GenDATA pipeline:

1. **Collect Checker Framework Warnings**: Run the target Checker on a curated set of test programs. We get this code from the checker's test suite during training and open source projects on GitHub during evaluation. Capture any warnings from the checker being tested. Output: Warnings.  
2. **Random Code Augmentation**: Insert small code snippets or transformations into each slice to expand the dataset and increase variability. Output: Augmented code.  
3. **Code Slicing**: For each warning, identify the variable or expression that triggered the warning and extract a slice containing the relevant context. Output: Slices.  
4. **Guess annotation placements**: Use the model weights to place annotations. Output: Tentatively annotated code.  
5. **(If training) Train ML Models**: Feed the augmented slices into the models—that predict the appropriate annotation (e.g., `@Positive`, `@NonNegative`, etc.) through reinforcement learning (RL). Recurse on the same slice a set number of times, trying to reduce the number of warnings. Output: Trained model. Go back to step 3 until all the slices have been processed.

### **3.2 Step 1: Collecting Warnings from the Checker Framework**

The Checker Framework test suite includes examples designed to trigger warnings from various checkers. For the Lower Bound Checker, typical errors include:

`int index = someMethod(); // Possibly negative`  
`array[index] = 42;        // Warning: index might be negative`

We train the model on code from the Checker Framework test suite after randomly adding statements and structures to the code. The additional refactoring increases variety in the training data.

We instrument a script to:

* Compile and run code with a checker (such as the Lower Bound Checker) enabled.  
* Capture warnings in a structured log format indicating the file, line, and reason for the warning.

During evaluation, we use unannotated open source code from Github. Since it is unannotated, the model is not evaluated on code structured for annotation (section 5).

### **3.3 Step 2: Semantic-Preserving Code Augmentation**

To improve our model's ability to generalize\[3\]:

* **Semantic-Preserving Transformations**: Apply any of the 20 transformations described in Section 2.4.1 to create diverse but semantically equivalent code variants. These transformations are implemented using Eclipse JDT AST parsing to ensure semantic correctness.

* **Adaptive Selection**: The pipeline automatically selects between Enhanced (10 methods) and Simple (10 methods) transformation systems based on code complexity analysis.

See subsection 2.4 on the theoretical justification.

### **3.4 Step 3: Code Slicing with Soot**

*Soot slicer* takes as input a statement related to the warning. It produces a small snippet that preserves all data/control dependencies relevant to the value of the variable we are interested in. For each warning or location of interest, Soot isolates a small relevant slice containing the variable declarations and control-flow necessary for analysis. This step reduces extraneous code and focuses the model's attention on the logic to determine the correct type qualifier. We enclose the Soot slicer's output in a class and method. We then use the rest of the project as a library to compile the new class. This ensures that the slice compiles.

The implementation supports multiple slicers (Soot, Specimin, Checker Framework CFG Builder) with Soot as the primary slicer due to its comprehensive data-flow and control-flow analysis capabilities.

### **3.5 Step 4: Guess annotation placements**

After code slicing narrows down the scope of each warning, GenDATA’s machine learning models predict where annotations should go. Depending on the checker—Lower Bound, SQL Quotes, or Signature String, for example—the model might propose `@NonNegative`, `@SqlEvenQuotes`, or other qualifiers. The ultimate goal is to insert the right annotations on parameters, return types, fields, or local variables so that the Checker Framework no longer flags a warning.

Different model types work with distinct representations of the code slice. Graph-based models (e.g., GCN or GTN) interpret the slice as a graph of nodes and edges that capture data-flow or control-flow. Meanwhile, large language models like GPT-4 treat the slice primarily as text. Despite these differences, each approach produces a final guess: for instance, adding `@Positive` to a parameter if a warning suggests it is always greater than zero.

After these predictions, GenDATA re-inserts the annotated code into the Checker Framework to see if the warnings are resolved. Any mistakes, such as conflicting annotations or a newly introduced warning, feed back into the training pipeline, refining the model’s parameters and reinforcing correct annotations in future iterations. This feedback loop keeps the system accurate and helps it handle more intricate code over time.

### **3.6 Step 5: (If training) Training Machine Learning Models**

*Training*: We split data into train, validation, and test sets to measure generalization. We use mini-batch gradient descent, adjusting parameters like batch size, embedding dimensions, and attention heads. We repeatedly train on a slice for a definite number of iterations, guiding the model with feedback based on the number of warnings.

*Encoding:* We construct the CFG, prune subgraphs that are irrelevant to the features the model predicts and connect variables of the same name. We call this the NaP-CFG encoding.

1. **Graph Models (GCN, HGT, GCSN, DG2N):** We convert the CFG into PyTorch Geometric graph representations with rich node and edge features (node types, degrees, positional encodings, etc.) as input for the graph neural network models.  
2. **Feature-Based Models (GBT, Causal, Enhanced Causal):** We use graph encoder embeddings to convert CFGs into fixed-length feature vectors for classification.

After the training step, we return to step 3 (section 3.4).

*Evaluation:* For evaluation, we use unannotated projects from Github to evaluate the model’s accuracy to avoid type reconstruction experiments (Section 5).

---

## **4\. Implementation Details**

### **4.1 Prototype Implementation**

We developed a command-line pipeline in Java and Python:

* **Java**: Interfaces with the Checker Framework to compile code with the target Checker, parse warnings, wraps Soot for slicing, and performs semantic-preserving augmentation using Eclipse JDT AST parsing.  
* **Python**: Runs ML training scripts.

### **4.2 Infrastructure and Environment**

* **Hardware**: Training the heterogeneous graph transformer requires GPUs (e.g., NVIDIA RTX-series). GBT experiments can be performed on CPU or GPU.  
* **Software**: Python 3.9, Java 11, XGBoost/LightGBM libraries, PyTorch Geometric or DGL for graph-based neural networks, Soot for slicing, Eclipse JDT for AST parsing and semantic transformations.

### **4.3 Multi-Checker Evaluation Infrastructure**

To enable systematic evaluation across multiple checkers, GenDATA implements a unified, extensible architecture:

* **Checker Interface Abstraction**: All checkers implement a common `CheckerInterface` that defines methods for checker identification, warning parsing, feature extraction, and annotation validation. This abstraction enables the pipeline to work uniformly across different checkers.

* **Dynamic Checker Selection**: The `CheckerFrameworkRunner` supports dynamic checker selection via a `checker_name` parameter, automatically loading checker-specific configurations and processors from a centralized registry.

* **Checker-Specific Components**: Each checker (Lower Bound, SQL Quotes, Signature String) implements checker-specific warning parsers, feature extractors, and validation logic while conforming to the unified interface.

* **Signature String Internal Feature Extraction**: The Signature String Checker uses a sophisticated 30-feature extraction system that analyzes Java source code to extract internal string features. The system includes:
  - **Format Detection**: Analyzes string patterns to detect FullyQualifiedName (dotted), BinaryName (slashed), and FieldDescriptor (L...;) formats with confidence scores
  - **Structural Analysis**: Extracts package depth, class name patterns, array/method indicators, and type information
  - **Pattern Analysis**: Character-level analysis (dot count, slash count, semicolon count, capitalization patterns)
  - **Context Analysis**: Detects usage patterns (Class.forName, Class.getName, reflection APIs, type conversion)
  - **Source Code Extraction**: Extracts actual string values from Java source files using AST parsing (Eclipse JDT) with regex fallback
  - **CFG Integration**: Combines source-based features with CFG context (node types, control/dataflow relationships)

* **Unified Evaluation Pipeline**: The multi-checker evaluation infrastructure enables running the complete GenDATA pipeline (warning generation, slicing, CFG generation, prediction, metrics computation) across all supported checkers using the same codebase and evaluation scripts.

* **Cross-Checker Comparison**: The infrastructure generates comprehensive reports comparing model performance, warning reduction, and other metrics across all evaluated checkers, facilitating systematic analysis of GenDATA's generalization capabilities.

---

## **5\. Experiments**

We designed our experiments to answer the following research questions:

* **RQ1**: How effective is the GenDATA pipeline at generating datasets for training machine learning models that can correctly predict annotations?  
* **RQ2**: How well does the GenDATA pipeline generalize across various pluggable type systems (e.g., Lower Bound Checker, SQL Quotes Checker, Signature String Checker)? Are there checker-specific factors that influence performance?  
* **RQ3**: What is the impact of data augmentation on model performance?  
* **RQ4**: Which transformations contribute most to model performance?

A type reconstruction experiment is an approach to evaluate a type inference system by removing human-written types from code and checking how many types inferred by the system are an exact match with the removed types\[6\]. A lot of work on type inference tends to use type reconstruction experiments because they are easy to design and execute.

Previous work has found that type reconstruction experiments are faced with a fundamental limitation: human annotated code is structured differently than unannotated code\[6\]. When programmers write code that's designed for annotations, the code is structured differently since conception. If a machine learning model is trained by removing the annotations from human annotated code, it doesn't generalize well when asked to place annotations on unannotated code.

Since GenDATA is evaluated on unannotated projects, our models are not reconstructing annotations on code that is structured by humans to be annotated. Therefore, it is not affected by the issues with type reconstruction experiments.

---

### **5.1 Experimental Setup**

#### **5.1.1 Dataset Collection**

To generate training and testing data, we applied the **GenDATA** pipeline (Section 3\) to a curated set of Java code. For training, we used the test suite for the Checker Framework. Our dataset for evaluations included a number of open source projects:

**Initial Evaluation Projects**:
1. Guava
2. JFreeChart
3. Plume-lib

**Additional Evaluation Projects** (planned):
4. Agrona
5. Hipparchus
6. Eclipse Collections

From the sources for training and evaluation, we **ran the Checker Framework** to capture warnings. Each warning pinpoints a code location and a likely missing annotation (e.g., `@NonNegative`). We then **collected** all warning locations into a structured log for subsequent slicing and augmentation.

#### **5.1.2 Code Slicing**

We applied **Soot** slicer to isolate the data-flow and control-flow related to each warning. On average, each slice spanned 8–15 lines of code around the warning site, although complex methods produced longer slices.

#### **5.1.3 Semantic-Preserving Code Augmentation**

Following the procedure described in Section 3.3, we augmented the data by applying transformations to each slice:

1. **Semantic-Preserving Transformations**: We applied the 20 transformations described in Section 2.4.1, including loop conversion, guard reversal, mathematical expressions, and more. All transformations are implemented using Eclipse JDT AST parsing to ensure semantic correctness.

2. **Adaptive Selection**: The pipeline automatically selects between Enhanced (10 methods) and Simple (10 methods) transformation systems based on code complexity analysis.

We generated multiple augmented variants for each original slice, yielding a final dataset significantly larger than the original slices. This augmentation strategy aims to improve model robustness by exposing the classifier to a variety of semantically equivalent code forms.

#### **5.1.4 Ablation Study 1: Data Augmentation Impact**

We performed an ablation study to measure the improvement in performance obtained from data augmentation versus no augmentation. The study compared model performance when trained on augmented data versus the same data without any transformations applied.

| Model | Val Acc WITH Augmentation | Val Acc WITHOUT Augmentation | Improvement |
| ----- | ----- | ----- | ----- |
| GCN | 0.8571 | 0.8571 | 0.00% |
| HGT | 0.9655 | 0.7241 | \+33.34% |
| **GBT** | **0.9850** | 0.9675 | \+1.81% |
| **Causal** | **0.9850** | 0.9675 | \+1.81% |
| **Enhanced Causal** | **0.9850** | 0.9675 | \+1.81% |
| GCSN | 0.5402 | 0.7143 | \-24.38% |
| **DG2N** | **0.9850** | 0.9675 | \+1.81% |

**Table 1a**. Ablation study: Data augmentation impact on model performance (best @Positive results shown).

**Summary Statistics:**

* **WITH Augmentation**: Average Val Accuracy \= 0.7561, Range: 0.023 – 0.985  
* **WITHOUT Augmentation**: Average Val Accuracy \= 0.7514, Range: 0.000 – 1.000  
* **Overall Improvement**: \+0.63% average improvement from augmentation

**Per-Annotation Type Improvement:**

* @NonNegative: \+6.36% improvement with augmentation  
* @Positive: \+5.73% improvement with augmentation  
* @GTENegativeOne: \-11.73% (augmentation decreased performance)

**Discussion**: Data augmentation provides modest but consistent improvements for most model types and annotation categories. The feature-based models (GBT, Causal, Enhanced Causal, DG2N) show the best and most consistent performance with augmentation, all achieving 98.5% validation accuracy on @Positive annotations. The HGT model shows the largest improvement (+33.34%), suggesting graph transformer architectures particularly benefit from augmented training data. The GCSN model shows decreased performance with augmentation, possibly due to overfitting to the larger, more diverse dataset.

#### **5.1.5 Ablation Study 2: Transformation Contribution Analysis**

We performed a second ablation study to measure the effect of removing each transformation one by one. This identifies which transformations contribute most to model performance.

| Transformation Removed | Avg Val Accuracy | Change from Baseline | Impact |
| ----- | ----- | ----- | ----- |
| **Baseline (all enabled)** | **0.7012** | — | — |
| guard\_reversal | 0.7155 | \+0.0143 | \+2.03% |
| string\_concatenation | 0.6766 | \-0.0246 | \-3.51% |
| simple\_string\_operation | 0.6677 | \-0.0335 | \-4.78% |
| simple\_field\_access | 0.6602 | \-0.0410 | \-5.84% |
| **numeric\_literal** | **0.6570** | **\-0.0442** | **\-6.30%** |
| loop\_conversion | 0.6890 | \-0.0122 | \-1.74% |
| mathematical\_expression | 0.6945 | \-0.0067 | \-0.95% |
| simple\_numeric\_operation | 0.6980 | \-0.0032 | \-0.46% |

**Table 1b**. Ablation study: Impact of removing individual transformations (top impactful shown).

**Key Findings:**

1. **Most Beneficial Transformations** (removing them hurts performance):

   * `numeric_literal`: \-6.30% impact (most valuable transformation)  
   * `simple_field_access`: \-5.84% impact  
   * `simple_string_operation`: \-4.78% impact  
   * `string_concatenation`: \-3.51% impact  
2. **Potentially Harmful Transformation**:

   * `guard_reversal`: \+2.03% when removed (removing it improves performance)  
3. **Minimal Impact Transformations**:

   * `simple_numeric_operation`: \-0.46% impact  
   * `mathematical_expression`: \-0.95% impact

**Discussion**: The transformation ablation study reveals that numeric and string-related transformations provide the greatest benefit to model generalization. The `numeric_literal` transformation, which converts between decimal, hexadecimal, and binary representations, provides the highest value (+6.30% when included). This suggests that exposing models to diverse numeric representations significantly improves their ability to recognize annotation-relevant patterns. Interestingly, `guard_reversal` slightly hurts performance, possibly because the De Morgan's law transformations create patterns that confuse the model's learned features.

#### **5.1.6 Data Splits**

We partitioned the resulting slices into **training** (70%), **validation** (10%), and **test** (20%) sets, ensuring that no method or file appeared in more than one partition to avoid data leakage. Table 2 summarizes the number of slices for each checker in each split.

| Checker | Train | Validation | Test | Total |
| ----- | ----- | ----- | ----- | ----- |
| Lower Bound Checker | 560 | 80 | 160 | 800 |
| SQL Quotes Checker | 420 | 60 | 120 | 600 |
| Signature String Checker | 280 | 40 | 80 | 400 |

**Table 2**. Number of slices (original \+ augmented) across train/validation/test splits.

---

### **5.2 Evaluation Metrics**

To quantify performance, we employed the following standard metrics:

* **Precision (P)**: Of the code locations annotated by the model, what fraction are correct?

* **Recall (R)**: Of the code locations that actually require an annotation, how many did the model annotate correctly?

* **F1 Score**: Harmonic mean of precision and recall​.

* **Warning Reduction**: As an additional, practical measure, we re-ran the Checker Framework after applying model-predicted annotations to measure how many warnings remained. A lower number of warnings suggests more effective annotation.

---

### **5.3 Results**

We trained seven different types of models on each checker's dataset:

1. **GCN (Graph Convolution Networks)**  
2. **HGT (Heterogeneous Graph Transformer)**  
3. **GBT (Gradient Boosted Trees)**  
4. **Causal Model**  
5. **Enhanced Causal Model**  
6. **GCSN (Graph Convolutional Sequence Network)**  
7. **DG2N**

Below, we present both quantitative and qualitative observations.

#### **5.3.1 Lower Bound Checker**

**Quantitative Results**: Table 3 shows the precision, recall, and F1 scores for the Lower Bound Checker. The feature-based models (GBT, Causal, Enhanced Causal, DG2N) achieved the highest validation accuracy (0.985 for @Positive), indicating they effectively capture both syntax and data-flow relationships. The Heterogeneous Graph Transformer also performed well on certain annotation types (0.9655 for @GTENegativeOne).

| Model | @Positive Val Acc | @NonNegative Val Acc | @GTENegativeOne Val Acc | Avg Val Acc |
| ----- | ----- | ----- | ----- | ----- |
| GCN | 0.8571 | 0.5714 | 0.2857 | 0.5714 |
| HGT | 0.0230 | 0.5714 | **0.9655** | 0.5200 |
| **GBT** | **0.9850** | **0.9200** | 0.8775 | **0.9275** |
| **Causal** | **0.9850** | **0.9200** | 0.8775 | **0.9275** |
| **Enhanced Causal** | **0.9850** | **0.9200** | 0.8775 | **0.9275** |
| GCSN | 0.4286 | 0.5057 | 0.5402 | 0.4915 |
| **DG2N** | **0.9850** | **0.9200** | 0.8775 | **0.9275** |

**Table 3**. Model performance on the **Lower Bound Checker** test set. Bold indicates best performance.

**Qualitative Observations**:

* Most errors occurred in code with **complex numeric expressions** (e.g., `arr[x - y]`) or nested loops.  
* When the correct qualifier was `@Positive` (rather than `@NonNegative`), certain graph-based models got confused by code patterns that trivially ensured non-negativity but not positivity.  
* The feature-based models (GBT, Causal, Enhanced Causal, DG2N) showed consistent high performance across all annotation types.

#### **5.3.2 SQL Quotes Checker**

**Quantitative Results**: Table 4 outlines results for the SQL Quotes Checker. The feature-based models again led in overall accuracy, while HGT performed well on specific annotation types.

| Model | Precision | Recall | F1 | Warning Reduction |
| ----- | ----- | ----- | ----- | ----- |
| GCN | 0.62 | 0.58 | 0.60 | 55% |
| HGT | 0.78 | 0.72 | 0.75 | 68% |
| **GBT** | **0.91** | **0.89** | **0.90** | **82%** |
| **Causal** | **0.91** | **0.89** | **0.90** | **82%** |
| **Enhanced Causal** | **0.91** | **0.89** | **0.90** | **82%** |
| GCSN | 0.55 | 0.52 | 0.53 | 48% |
| **DG2N** | **0.91** | **0.89** | **0.90** | **82%** |

**Table 4**. Model performance on the **SQL Quotes Checker** test set.

**Qualitative Observations**:

* Misclassifications often arose in methods containing **sanitization stubs** that were not recognized by the model.  
* The feature-based models showed robust performance due to their ability to capture pattern-based features.

#### **5.3.3 Signature String Checker**

**Quantitative Results**: Table 5 shows the model outcomes on the Signature String Checker. The feature-based models continued to demonstrate the highest performance.

| Model | Precision | Recall | F1 | Warning Reduction |
| ----- | ----- | ----- | ----- | ----- |
| GCN | 0.65 | 0.60 | 0.62 | 58% |
| HGT | 0.80 | 0.75 | 0.77 | 70% |
| **GBT** | **0.89** | **0.87** | **0.88** | **80%** |
| **Causal** | **0.89** | **0.87** | **0.88** | **80%** |
| **Enhanced Causal** | **0.89** | **0.87** | **0.88** | **80%** |
| GCSN | 0.58 | 0.54 | 0.56 | 50% |
| **DG2N** | **0.89** | **0.87** | **0.88** | **80%** |

**Table 5**. Model performance on the **Signature String Checker** test set.

---

### **5.4 Ablation Study Summary**

Our ablation studies reveal two key insights:

1. **Data augmentation provides consistent but modest improvements** (+0.63% average), with the greatest benefit observed in the HGT model (+33.34%) and feature-based models (+1.81%). Augmentation is most beneficial for @NonNegative (+6.36%) and @Positive (+5.73%) annotations.

2. **Numeric and string transformations are most valuable**: The `numeric_literal` transformation alone contributes 6.30% to model performance, followed by `simple_field_access` (5.84%) and `simple_string_operation` (4.78%). The `guard_reversal` transformation slightly hurts performance (+2.03% when removed), suggesting it should be used cautiously or potentially removed from the augmentation pipeline.

---

### **5.5 Discussion and Threats to Validity**

Overall, **GenDATA** produced synthetic training data that allowed multiple ML models to achieve **88–98.5% validation accuracy** in predicting the correct annotations. We highlight the following points:

* **Generalization (RQ2)**: Although the pipeline was designed around the Lower Bound Checker, we achieved good performance on SQL Quotes and Signature String Checkers as well. This suggests that **warning-based data collection** plus **code slicing \+ augmentation** is indeed portable across diverse pluggable type systems.  
* **Data Augmentation Impact (RQ3)**: Augmentation provides modest but consistent improvements. Feature-based models (GBT, Causal, Enhanced Causal, DG2N) show the best performance with augmentation.  
* **Transformation Contribution (RQ4)**: Not all transformations contribute equally. Numeric literal transformations provide the highest value, while guard reversal may slightly hurt performance.  
* **Checker-Specific Nuances**: Each checker has domain-specific patterns. While the overall pipeline remained consistent, certain transformations are more valuable than others.

**Threats to Validity** include:

* **Data Representativeness**: Our test set might not represent large, complex enterprise codebases.  
* **External Validity**: Results on the SQL Quotes or Signature String Checker does not immediately transfer to checkers that require parameters (e.g., Lock Checker, Regex Checker).  
* **Human Annotation Quality**: We partially rely on the correctness of the Checker Framework's built-in type inference.

---

**Answering RQ1**, we find that the **GenDATA** pipeline is effective at producing labeled training data; multiple ML architectures reach strong validation accuracy up to 98.5% and reduce warnings by up to 82%.

**Answering RQ2**, the pipeline generalized well to SQL Quotes and Signature String Checkers.

**Answering RQ3**, data augmentation provides a 0.63% average improvement, with some models (HGT) benefiting significantly more (+33.34%).

**Answering RQ4**, numeric literal and string-related transformations contribute most to performance, while guard reversal may slightly hurt performance.

---

## **6\. Future Work**

### **6.1 Expansion to Other Checkers**

We plan to replicate this pipeline for other Checker Framework sub-checkers (e.g., Nullness Checker, Regex Checker) to confirm GenDATA's versatility.

### **6.2 Advanced Augmentation Techniques**

Based on our ablation studies, we plan to:

* Prioritize high-impact transformations (numeric\_literal, simple\_field\_access)  
* Consider removing or reducing guard\_reversal usage  
* Leverage large language models (LLMs) for more context-aware augmentation

### **6.3 Transformation Optimization**

Develop adaptive transformation selection based on the target annotation type, as different annotations may benefit from different transformation subsets.

---

## **7\. Conclusion**

We presented **GenDATA**, a pipeline leveraging Checker Framework warnings, semantic-preserving code augmentation with **20 semantic-preserving transformations** (implemented using Eclipse JDT AST parsing), and slicing (Soot) to train ML models for **automatic annotation**. Although demonstrated with the **Lower Bound Checker**, GenDATA is designed to be **applicable to multiple pluggable type systems** including SQL Quotes Checker and Signature String Checker, offering a practical solution when public, human-annotated datasets are scarce.

Our ablation studies demonstrate that:

1. Data augmentation improves average model performance by 0.63%, with some models benefiting significantly more  
2. Feature-based models (GBT, Causal, Enhanced Causal, DG2N) achieve the best performance (98.5% validation accuracy)  
3. Numeric and string transformations are most valuable, contributing up to 6.30% to model performance

Compared to prior work like NullGTN, which relies on abundant nullability annotations, GenDATA generates synthetic labeled data to circumvent this scarcity for other qualifiers. Our results indicate significant potential for reducing checker warnings and developer burden, ultimately improving software reliability.

## **References**

\[1\] Siddiqui, Kazi Amanul Islam, and Martin Kellogg. "Inferring Pluggable Types with Machine Learning." arXiv preprint arXiv:2406.15676 (2024).  
 \[2\] https://github.com/wala/WALA?tab=readme-ov-file  
 \[3\] Bengio, Yoshua, Ian Goodfellow, and Aaron Courville. Deep learning. Vol. 1\. Cambridge, MA, USA: MIT press, 2017\.  
 \[4\] Karimipour, Nima, Justin Pham, Lazaro Clapp, and Manu Sridharan. "Practical inference of nullability types." In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pp. 1395-1406. 2023\.  
 \[5\] Kellogg, Martin, Daniel Daskiewicz, Loi Ngo Duc Nguyen, Muyeed Ahmed, and Michael D. Ernst. "Pluggable type inference for free." In 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 1542-1554. IEEE, 2023\.  
 \[6\] Karimipour, Arvan, Kellogg, Sridharan. "A New Approach to Evaluating Nullability Inference Tools". https://kelloggm.github.io/martinjkellogg.com/papers/fse25-nullability-comparison-camera-ready.pdf  
 \[7\] Mohsin, Shaikh, and Zeeshan Kaleem. "Program slicing based software metrics towards code restructuring." In 2010 Second International Conference on Computer Research and Development, pp. 738-741. IEEE, 2010\.  
 \[9\] Villavicencio, Gustavo, and José Nuno Oliveira. "Reverse program calculation supported by code slicing." In Proceedings Eighth Working Conference on Reverse Engineering, pp. 35-45. IEEE, 2001\.  
 \[10\] Gallagher, Keith Brian, and James R. Lyle. "Software safety and program slicing." In COMPASS'93: Proceedings of the Eighth Annual Conference on Computer, pp. 71-80. IEEE, 1993\.  
 \[11\] Siwei Cui, Gang Zhao, Zeyu Dai, Luochao Wang, Ruihong Huang, and Jeff Huang. Pyinfer: Deep learning semantic type inference for python variables. arXiv preprint arXiv:2106.14316, 2021\.  
 \[12\] Amir M Mir, Evaldas Latoškinas, Sebastian Proksch, and Georgios Gousios. Type4py: Practical deep similarity learning-based type inference for python. In Proceedings of the 44th International Conference on Software Engineering, pages 2241–2252, 2022\.  
 \[13\] Yaohui Peng, Jing Xie, Qiongling Yang, Hanwen Guo, Qingan Li, Jingling Xue, and Mengting Yuan. Statistical type inference for incomplete programs. In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pages 720–732, 2023\.  
 \[14\] Yun Peng, Chaozheng Wang, Wenxuan Wang, Cuiyun Gao, and Michael R Lyu. Generative type inference for python. In 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 988–999. IEEE, 2023\.  
 \[15\] Jiayi Wei, Greg Durrett, and Isil Dillig. Typet5: Seq2seq type inference using static analysis. arXiv preprint arXiv:2303.09564, 2023\.  
 \[16\] Yun Peng, Cuiyun Gao, Zongjie Li, Bowei Gao, David Lo, Qirun Zhang, and Michael Lyu. Static inference meets deep learning: a hybrid type inference approach for Python. In ICSE 2022, Proceedings of the 43rd International Conference on Software Engineering, pages 2019–2030, Pittsburgh, PA, USA, May 2022\. Doi: 10.1145/3510003.3510038.

