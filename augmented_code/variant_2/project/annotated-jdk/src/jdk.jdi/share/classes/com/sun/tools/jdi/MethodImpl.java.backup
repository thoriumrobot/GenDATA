/*
 * Copyright (c) 1998, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package com.sun.tools.jdi;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.ArrayList;
import java.util.List;
import com.sun.jdi.AbsentInformationException;
import com.sun.jdi.ArrayReference;
import com.sun.jdi.ArrayType;
import com.sun.jdi.ClassNotLoadedException;
import com.sun.jdi.InterfaceType;
import com.sun.jdi.InvalidTypeException;
import com.sun.jdi.Location;
import com.sun.jdi.Method;
import com.sun.jdi.Type;
import com.sun.jdi.Value;
import com.sun.jdi.VirtualMachine;

public abstract class MethodImpl extends TypeComponentImpl implements Method {

    abstract int argSlotCount() throws AbsentInformationException;

    abstract List<Location> allLineLocations(SDE.Stratum stratum, String sourceName) throws AbsentInformationException;

    abstract List<Location> locationsOfLine(SDE.Stratum stratum, String sourceName, int lineNumber) throws AbsentInformationException;

    static MethodImpl createMethodImpl(VirtualMachine vm, ReferenceTypeImpl declaringType, long ref, String name, String signature, String genericSignature, int modifiers);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public final List<Location> allLineLocations() throws AbsentInformationException;

    public List<Location> allLineLocations(String stratumID, String sourceName) throws AbsentInformationException;

    public final List<Location> locationsOfLine(int lineNumber) throws AbsentInformationException;

    public List<Location> locationsOfLine(String stratumID, String sourceName, int lineNumber) throws AbsentInformationException;

    LineInfo codeIndexToLineInfo(SDE.Stratum stratum, long codeIndex);

    public String returnTypeName();

    public Type returnType() throws ClassNotLoadedException;

    public Type findType(String signature) throws ClassNotLoadedException;

    public List<String> argumentTypeNames();

    public List<String> argumentSignatures();

    Type argumentType(int index) throws ClassNotLoadedException;

    public List<Type> argumentTypes() throws ClassNotLoadedException;

    public int compareTo(Method method);

    public boolean isAbstract();

    public boolean isDefault();

    public boolean isSynchronized();

    public boolean isNative();

    public boolean isVarArgs();

    public boolean isBridge();

    public boolean isConstructor();

    public boolean isStaticInitializer();

    public boolean isObsolete();

    class ReturnContainer implements ValueContainer {

        public Type type() throws ClassNotLoadedException;

        public String typeName();

        public String signature();

        public Type findType(String signature) throws ClassNotLoadedException;
    }

    ReturnContainer getReturnValueContainer();

    class ArgumentContainer implements ValueContainer {

        public Type type() throws ClassNotLoadedException;

        public String typeName();

        public String signature();

        public Type findType(String signature) throws ClassNotLoadedException;
    }

    void handleVarArgs(List<Value> arguments) throws ClassNotLoadedException, InvalidTypeException;

    List<Value> validateAndPrepareArgumentsForInvoke(List<? extends Value> origArguments) throws ClassNotLoadedException, InvalidTypeException;

    public String toString();
}
