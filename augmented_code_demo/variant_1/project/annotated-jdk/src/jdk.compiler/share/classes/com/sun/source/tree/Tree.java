/*
    @Positive
 * Copyright (c) 2005, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package com.sun.source.tree;

    @Positive
import jdk.internal.javac.PreviewFeature;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public interface Tree {

    @Positive
    public enum Kind {

    @Positive
        ANNOTATED_TYPE(AnnotatedTypeTree.class),
    @Positive
        ANNOTATION(AnnotationTree.class),
    @Positive
        TYPE_ANNOTATION(AnnotationTree.class),
    @Positive
        ARRAY_ACCESS(ArrayAccessTree.class),
    @Positive
        ARRAY_TYPE(ArrayTypeTree.class),
    @Positive
        ASSERT(AssertTree.class),
    @Positive
        ASSIGNMENT(AssignmentTree.class),
    @Positive
        BLOCK(BlockTree.class),
    @Positive
        BREAK(BreakTree.class),
    @Positive
        CASE(CaseTree.class),
    @Positive
        CATCH(CatchTree.class),
    @Positive
        CLASS(ClassTree.class),
    @Positive
        COMPILATION_UNIT(CompilationUnitTree.class),
    @Positive
        CONDITIONAL_EXPRESSION(ConditionalExpressionTree.class),
    @Positive
        CONTINUE(ContinueTree.class),
    @Positive
        DO_WHILE_LOOP(DoWhileLoopTree.class),
    @Positive
        ENHANCED_FOR_LOOP(EnhancedForLoopTree.class),
    @Positive
        EXPRESSION_STATEMENT(ExpressionStatementTree.class),
    @Positive
        MEMBER_SELECT(MemberSelectTree.class),
    @Positive
        MEMBER_REFERENCE(MemberReferenceTree.class),
    @Positive
        FOR_LOOP(ForLoopTree.class),
    @Positive
        IDENTIFIER(IdentifierTree.class),
    @Positive
        IF(IfTree.class),
    @Positive
        IMPORT(ImportTree.class),
    @Positive
        INSTANCE_OF(InstanceOfTree.class),
    @Positive
        LABELED_STATEMENT(LabeledStatementTree.class),
    @Positive
        METHOD(MethodTree.class),
    @Positive
        METHOD_INVOCATION(MethodInvocationTree.class),
    @Positive
        MODIFIERS(ModifiersTree.class),
    @Positive
        NEW_ARRAY(NewArrayTree.class),
    @Positive
        NEW_CLASS(NewClassTree.class),
    @Positive
        LAMBDA_EXPRESSION(LambdaExpressionTree.class),
    @Positive
        PACKAGE(PackageTree.class),
    @Positive
        PARENTHESIZED(ParenthesizedTree.class),
    @Positive
        BINDING_PATTERN(BindingPatternTree.class),
    @Positive
        @PreviewFeature(feature = PreviewFeature.Feature.SWITCH_PATTERN_MATCHING, reflective = true)
    @Positive
        GUARDED_PATTERN(GuardedPatternTree.class),
    @Positive
        @PreviewFeature(feature = PreviewFeature.Feature.SWITCH_PATTERN_MATCHING, reflective = true)
    @Positive
        PARENTHESIZED_PATTERN(ParenthesizedPatternTree.class),
    @Positive
        @PreviewFeature(feature = PreviewFeature.Feature.SWITCH_PATTERN_MATCHING, reflective = true)
    @Positive
        DEFAULT_CASE_LABEL(DefaultCaseLabelTree.class),
    @Positive
        PRIMITIVE_TYPE(PrimitiveTypeTree.class),
    @Positive
        RETURN(ReturnTree.class),
    @Positive
        EMPTY_STATEMENT(EmptyStatementTree.class),
    @Positive
        SWITCH(SwitchTree.class),
    @Positive
        SWITCH_EXPRESSION(SwitchExpressionTree.class),
    @Positive
        SYNCHRONIZED(SynchronizedTree.class),
    @Positive
        THROW(ThrowTree.class),
    @Positive
        TRY(TryTree.class),
    @Positive
        PARAMETERIZED_TYPE(ParameterizedTypeTree.class),
    @Positive
        UNION_TYPE(UnionTypeTree.class),
    @Positive
        INTERSECTION_TYPE(IntersectionTypeTree.class),
    @Positive
        TYPE_CAST(TypeCastTree.class),
    @Positive
        TYPE_PARAMETER(TypeParameterTree.class),
    @Positive
        VARIABLE(VariableTree.class),
    @Positive
        WHILE_LOOP(WhileLoopTree.class),
    @Positive
        POSTFIX_INCREMENT(UnaryTree.class),
    @Positive
        POSTFIX_DECREMENT(UnaryTree.class),
    @Positive
        PREFIX_INCREMENT(UnaryTree.class),
    @Positive
        PREFIX_DECREMENT(UnaryTree.class),
    @Positive
        UNARY_PLUS(UnaryTree.class),
    @Positive
        UNARY_MINUS(UnaryTree.class),
    @Positive
        BITWISE_COMPLEMENT(UnaryTree.class),
    @Positive
        LOGICAL_COMPLEMENT(UnaryTree.class),
    @Positive
        MULTIPLY(BinaryTree.class),
    @Positive
        DIVIDE(BinaryTree.class),
    @Positive
        REMAINDER(BinaryTree.class),
    @Positive
        PLUS(BinaryTree.class),
    @Positive
        MINUS(BinaryTree.class),
    @Positive
        LEFT_SHIFT(BinaryTree.class),
    @Positive
        RIGHT_SHIFT(BinaryTree.class),
    @Positive
        UNSIGNED_RIGHT_SHIFT(BinaryTree.class),
    @Positive
        LESS_THAN(BinaryTree.class),
    @Positive
        GREATER_THAN(BinaryTree.class),
    @Positive
        LESS_THAN_EQUAL(BinaryTree.class),
    @Positive
        GREATER_THAN_EQUAL(BinaryTree.class),
    @Positive
        EQUAL_TO(BinaryTree.class),
    @Positive
        NOT_EQUAL_TO(BinaryTree.class),
    @Positive
        AND(BinaryTree.class),
    @Positive
        XOR(BinaryTree.class),
    @Positive
        OR(BinaryTree.class),
    @Positive
        CONDITIONAL_AND(BinaryTree.class),
    @Positive
        CONDITIONAL_OR(BinaryTree.class),
    @Positive
        MULTIPLY_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        DIVIDE_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        REMAINDER_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        PLUS_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        MINUS_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        LEFT_SHIFT_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        RIGHT_SHIFT_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        UNSIGNED_RIGHT_SHIFT_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        AND_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        XOR_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        OR_ASSIGNMENT(CompoundAssignmentTree.class),
    @Positive
        INT_LITERAL(LiteralTree.class),
    @Positive
        LONG_LITERAL(LiteralTree.class),
    @Positive
        FLOAT_LITERAL(LiteralTree.class),
    @Positive
        DOUBLE_LITERAL(LiteralTree.class),
    @Positive
        BOOLEAN_LITERAL(LiteralTree.class),
    @Positive
        CHAR_LITERAL(LiteralTree.class),
    @Positive
        STRING_LITERAL(LiteralTree.class),
    @Positive
        NULL_LITERAL(LiteralTree.class),
    @Positive
        UNBOUNDED_WILDCARD(WildcardTree.class),
    @Positive
        EXTENDS_WILDCARD(WildcardTree.class),
    @Positive
        SUPER_WILDCARD(WildcardTree.class),
    @Positive
        ERRONEOUS(ErroneousTree.class),
    @Positive
        INTERFACE(ClassTree.class),
    @Positive
        ENUM(ClassTree.class),
    @Positive
        ANNOTATION_TYPE(ClassTree.class),
    @Positive
        MODULE(ModuleTree.class),
    @Positive
        EXPORTS(ExportsTree.class),
    @Positive
        OPENS(OpensTree.class),
    @Positive
        PROVIDES(ProvidesTree.class),
    @Positive
        RECORD(ClassTree.class),
    @Positive
        REQUIRES(RequiresTree.class),
    @Positive
        USES(UsesTree.class),
    @Positive
        OTHER(null),
    @Positive
        YIELD(YieldTree.class);

    @Positive
        Kind(Class<? extends Tree> intf) {
    @Positive
        }

    @Positive
        public Class<? extends Tree> asInterface();

    @Positive
        private final Class<? extends Tree> associatedInterface;
    @Positive
    }

    @Positive
    @Pure
    @Positive
    Kind getKind();

    @Positive
    <R, D> R accept(TreeVisitor<R, D> visitor, D data);
    @Positive
}

// CFWR semantic augmentation - variant 1
