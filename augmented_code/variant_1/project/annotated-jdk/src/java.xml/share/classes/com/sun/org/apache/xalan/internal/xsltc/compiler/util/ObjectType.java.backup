/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xalan.internal.xsltc.compiler.util;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.org.apache.bcel.internal.generic.ALOAD;
import com.sun.org.apache.bcel.internal.generic.ASTORE;
import com.sun.org.apache.bcel.internal.generic.BranchHandle;
import com.sun.org.apache.bcel.internal.generic.ConstantPoolGen;
import com.sun.org.apache.bcel.internal.generic.GOTO;
import com.sun.org.apache.bcel.internal.generic.IFNULL;
import com.sun.org.apache.bcel.internal.generic.INVOKEVIRTUAL;
import com.sun.org.apache.bcel.internal.generic.Instruction;
import com.sun.org.apache.bcel.internal.generic.InstructionList;
import com.sun.org.apache.bcel.internal.generic.PUSH;
import com.sun.org.apache.xalan.internal.utils.ObjectFactory;
import com.sun.org.apache.xalan.internal.xsltc.compiler.Constants;

public final class ObjectType extends Type {

    protected ObjectType(String javaClassName) {
    }

    protected ObjectType(Class<?> clazz) {
    }

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public String getJavaClassName();

    public Class<?> getJavaClass();

    public String toString();

    public boolean identicalTo(Type other);

    public String toSignature();

    public com.sun.org.apache.bcel.internal.generic.Type toJCType();

    public void translateTo(ClassGenerator classGen, MethodGenerator methodGen, Type type);

    public void translateTo(ClassGenerator classGen, MethodGenerator methodGen, StringType type);

    public void translateTo(ClassGenerator classGen, MethodGenerator methodGen, Class<?> clazz);

    public void translateFrom(ClassGenerator classGen, MethodGenerator methodGen, Class<?> clazz);

    public Instruction LOAD(int slot);

    public Instruction STORE(int slot);
}
