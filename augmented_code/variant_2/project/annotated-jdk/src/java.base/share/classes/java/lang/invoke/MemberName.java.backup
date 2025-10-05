/*
 * Copyright (c) 2008, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.invoke;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import sun.invoke.util.BytecodeDescriptor;
import sun.invoke.util.VerifyAccess;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import static java.lang.invoke.MethodHandleNatives.Constants.*;
import static java.lang.invoke.MethodHandleStatics.newIllegalArgumentException;
import static java.lang.invoke.MethodHandleStatics.newInternalError;

final class ResolvedMethodName {
}

final class MemberName implements Member, Cloneable {

    public Class<?> getDeclaringClass();

    public ClassLoader getClassLoader();

    public String getName();

    public MethodType getMethodOrFieldType();

    public MethodType getMethodType();

    String getMethodDescriptor();

    public MethodType getInvocationType();

    public Class<?>[] getParameterTypes();

    public Class<?> getReturnType();

    public Class<?> getFieldType();

    public Object getType();

    public String getSignature();

    public int getModifiers();

    public byte getReferenceKind();

    boolean referenceKindIsConsistentWith(int originalRefKind);

    public boolean isMethodHandleInvoke();

    public static boolean isMethodHandleInvokeName(String name);

    public boolean isVarHandleMethodInvoke();

    public static boolean isVarHandleMethodInvokeName(String name);

    public boolean isStatic();

    public boolean isPublic();

    public boolean isPrivate();

    public boolean isProtected();

    public boolean isFinal();

    public boolean canBeStaticallyBound();

    public boolean isVolatile();

    public boolean isAbstract();

    public boolean isNative();

    public boolean isBridge();

    public boolean isVarargs();

    public boolean isSynthetic();

    public boolean isInvocable();

    public boolean isFieldOrMethod();

    public boolean isMethod();

    public boolean isConstructor();

    public boolean isField();

    public boolean isType();

    public boolean isPackage();

    public boolean isCallerSensitive();

    public boolean isTrustedFinalField();

    public boolean isAccessibleFrom(Class<?> lookupClass);

    public boolean refersTo(Class<?> declc, String n);

    public MemberName(Method m) {
    }

    @SuppressWarnings("LeakingThisInConstructor")
    public MemberName(Method m, boolean wantSpecial) {
    }

    public MemberName asSpecial();

    public MemberName asConstructor();

    public MemberName asNormalOriginal();

    @SuppressWarnings("LeakingThisInConstructor")
    public MemberName(Constructor<?> ctor) {
    }

    public MemberName(Field fld) {
    }

    @SuppressWarnings("LeakingThisInConstructor")
    public MemberName(Field fld, boolean makeSetter) {
    }

    public boolean isGetter();

    public boolean isSetter();

    public MemberName asSetter();

    public MemberName(Class<?> type) {
    }

    static MemberName makeMethodHandleInvoke(String name, MethodType type);

    static MemberName makeMethodHandleInvoke(String name, MethodType type, int mods);

    static MemberName makeVarHandleMethodInvoke(String name, MethodType type);

    static MemberName makeVarHandleMethodInvoke(String name, MethodType type, int mods);

    @Override
    protected MemberName clone();

    public MemberName getDefinition();

    @Override
    @SuppressWarnings({ "deprecation", "removal" })
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object that);

    public boolean equals(MemberName that);

    public MemberName(Class<?> defClass, String name, Class<?> type, byte refKind) {
    }

    public MemberName(Class<?> defClass, String name, MethodType type, byte refKind) {
    }

    public MemberName(byte refKind, Class<?> defClass, String name, Object type) {
    }

    public boolean hasReceiverTypeDispatch();

    public boolean isResolved();

    void initResolved(boolean isResolved);

    void checkForTypeAlias(Class<?> refc);

    @SuppressWarnings("LocalVariableHidesMemberVariable")
    @Override
    public String toString();

    public IllegalAccessException makeAccessException(String message, Object from);

    public ReflectiveOperationException makeAccessException();

    static Factory getFactory();

    static class Factory {

        List<MemberName> getMembers(Class<?> defc, String matchName, Object matchType, int matchFlags, Class<?> lookupClass);

        public <NoSuchMemberException extends ReflectiveOperationException> MemberName resolveOrFail(byte refKind, MemberName m, Class<?> lookupClass, int allowedModes, Class<NoSuchMemberException> nsmClass) throws IllegalAccessException, NoSuchMemberException;

        public MemberName resolveOrNull(byte refKind, MemberName m, Class<?> lookupClass, int allowedModes);

        public List<MemberName> getMethods(Class<?> defc, boolean searchSupers, Class<?> lookupClass);

        public List<MemberName> getMethods(Class<?> defc, boolean searchSupers, String name, MethodType type, Class<?> lookupClass);

        public List<MemberName> getConstructors(Class<?> defc, Class<?> lookupClass);

        public List<MemberName> getFields(Class<?> defc, boolean searchSupers, Class<?> lookupClass);

        public List<MemberName> getFields(Class<?> defc, boolean searchSupers, String name, Class<?> type, Class<?> lookupClass);

        public List<MemberName> getNestedTypes(Class<?> defc, boolean searchSupers, Class<?> lookupClass);
    }
}
