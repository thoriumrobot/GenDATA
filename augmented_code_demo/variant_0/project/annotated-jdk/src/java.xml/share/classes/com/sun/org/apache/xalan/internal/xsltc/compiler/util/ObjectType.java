/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xalan.internal.xsltc.compiler.util;

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
import com.sun.org.apache.bcel.internal.generic.ALOAD;
    @Positive
import com.sun.org.apache.bcel.internal.generic.ASTORE;
    @Positive
import com.sun.org.apache.bcel.internal.generic.BranchHandle;
    @Positive
import com.sun.org.apache.bcel.internal.generic.ConstantPoolGen;
    @Positive
import com.sun.org.apache.bcel.internal.generic.GOTO;
    @Positive
import com.sun.org.apache.bcel.internal.generic.IFNULL;
    @Positive
import com.sun.org.apache.bcel.internal.generic.INVOKEVIRTUAL;
    @Positive
import com.sun.org.apache.bcel.internal.generic.Instruction;
    @Positive
import com.sun.org.apache.bcel.internal.generic.InstructionList;
    @Positive
import com.sun.org.apache.bcel.internal.generic.PUSH;
    @Positive
import com.sun.org.apache.xalan.internal.utils.ObjectFactory;
    @Positive
import com.sun.org.apache.xalan.internal.xsltc.compiler.Constants;

    @Positive
public final class ObjectType extends Type {

    @Positive
    protected ObjectType(String javaClassName) {
    @Positive
    }

    @Positive
    protected ObjectType(Class<?> clazz) {
    @Positive
    }

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public String getJavaClassName();

    @Positive
    public Class<?> getJavaClass();

    @Positive
    public String toString();

    @Positive
    public boolean identicalTo(Type other);

    @Positive
    public String toSignature();

    @Positive
    public com.sun.org.apache.bcel.internal.generic.Type toJCType();

    @Positive
    public void translateTo(ClassGenerator classGen, MethodGenerator methodGen, Type type);

    @Positive
    public void translateTo(ClassGenerator classGen, MethodGenerator methodGen, StringType type);

    @Positive
    public void translateTo(ClassGenerator classGen, MethodGenerator methodGen, Class<?> clazz);

    @Positive
    public void translateFrom(ClassGenerator classGen, MethodGenerator methodGen, Class<?> clazz);

    @Positive
    public Instruction LOAD(int slot);

    @Positive
    public Instruction STORE(int slot);
    @Positive
}

// CFWR semantic augmentation - variant 0
