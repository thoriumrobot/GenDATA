/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xalan.internal.xsltc.compiler;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import com.sun.org.apache.xalan.internal.xsltc.compiler.util.Type;
    @Positive
import com.sun.org.apache.xalan.internal.xsltc.compiler.util.TypeCheckError;
    @Positive
import java.util.Objects;

    @Positive
class VariableRefBase extends Expression {

    @Positive
    protected VariableBase _variable;

    @Positive
    protected Closure _closure;

    @Positive
    public VariableRefBase(VariableBase variable) {
    @Positive
    }

    @Positive
    public VariableRefBase() {
    @Positive
    }

    @Positive
    public VariableBase getVariable();

    @Positive
    public void addParentDependency();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public Type typeCheck(SymbolTable stable) throws TypeCheckError;
    @Positive
}

// CFWR semantic augmentation - variant 0
