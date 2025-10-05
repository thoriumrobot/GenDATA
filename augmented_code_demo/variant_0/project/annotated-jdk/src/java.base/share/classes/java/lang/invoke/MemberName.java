/*
    @Positive
 * Copyright (c) 2008, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.invoke;

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
import sun.invoke.util.BytecodeDescriptor;
    @Positive
import sun.invoke.util.VerifyAccess;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.lang.reflect.Member;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import static java.lang.invoke.MethodHandleNatives.Constants.*;
    @Positive
import static java.lang.invoke.MethodHandleStatics.newIllegalArgumentException;
    @Positive
import static java.lang.invoke.MethodHandleStatics.newInternalError;

    @Positive
final class ResolvedMethodName {
    @Positive
}

    @Positive
final class MemberName implements Member, Cloneable {

    @Positive
    public Class<?> getDeclaringClass();

    @Positive
    public ClassLoader getClassLoader();

    @Positive
    public String getName();

    @Positive
    public MethodType getMethodOrFieldType();

    @Positive
    public MethodType getMethodType();

    @Positive
    String getMethodDescriptor();

    @Positive
    public MethodType getInvocationType();

    @Positive
    public Class<?>[] getParameterTypes();

    @Positive
    public Class<?> getReturnType();

    @Positive
    public Class<?> getFieldType();

    @Positive
    public Object getType();

    @Positive
    public String getSignature();

    @Positive
    public int getModifiers();

    @Positive
    public byte getReferenceKind();

    @Positive
    boolean referenceKindIsConsistentWith(int originalRefKind);

    @Positive
    public boolean isMethodHandleInvoke();

    @Positive
    public static boolean isMethodHandleInvokeName(String name);

    @Positive
    public boolean isVarHandleMethodInvoke();

    @Positive
    public static boolean isVarHandleMethodInvokeName(String name);

    @Positive
    public boolean isStatic();

    @Positive
    public boolean isPublic();

    @Positive
    public boolean isPrivate();

    @Positive
    public boolean isProtected();

    @Positive
    public boolean isFinal();

    @Positive
    public boolean canBeStaticallyBound();

    @Positive
    public boolean isVolatile();

    @Positive
    public boolean isAbstract();

    @Positive
    public boolean isNative();

    @Positive
    public boolean isBridge();

    @Positive
    public boolean isVarargs();

    @Positive
    public boolean isSynthetic();

    @Positive
    public boolean isInvocable();

    @Positive
    public boolean isFieldOrMethod();

    @Positive
    public boolean isMethod();

    @Positive
    public boolean isConstructor();

    @Positive
    public boolean isField();

    @Positive
    public boolean isType();

    @Positive
    public boolean isPackage();

    @Positive
    public boolean isCallerSensitive();

    @Positive
    public boolean isTrustedFinalField();

    @Positive
    public boolean isAccessibleFrom(Class<?> lookupClass);

    @Positive
    public boolean refersTo(Class<?> declc, String n);

    @Positive
    public MemberName(Method m) {
    @Positive
    }

    @Positive
    @SuppressWarnings("LeakingThisInConstructor")
    @Positive
    public MemberName(Method m, boolean wantSpecial) {
    @Positive
    }

    @Positive
    public MemberName asSpecial();

    @Positive
    public MemberName asConstructor();

    @Positive
    public MemberName asNormalOriginal();

    @Positive
    @SuppressWarnings("LeakingThisInConstructor")
    @Positive
    public MemberName(Constructor<?> ctor) {
    @Positive
    }

    @Positive
    public MemberName(Field fld) {
    @Positive
    }

    @Positive
    @SuppressWarnings("LeakingThisInConstructor")
    @Positive
    public MemberName(Field fld, boolean makeSetter) {
    @Positive
    }

    @Positive
    public boolean isGetter();

    @Positive
    public boolean isSetter();

    @Positive
    public MemberName asSetter();

    @Positive
    public MemberName(Class<?> type) {
    @Positive
    }

    @Positive
    static MemberName makeMethodHandleInvoke(String name, MethodType type);

    @Positive
    static MemberName makeMethodHandleInvoke(String name, MethodType type, int mods);

    @Positive
    static MemberName makeVarHandleMethodInvoke(String name, MethodType type);

    @Positive
    static MemberName makeVarHandleMethodInvoke(String name, MethodType type, int mods);

    @Positive
    @Override
    @Positive
    protected MemberName clone();

    @Positive
    public MemberName getDefinition();

    @Positive
    @Override
    @Positive
    @SuppressWarnings({ "deprecation", "removal" })
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object that);

    @Positive
    public boolean equals(MemberName that);

    @Positive
    public MemberName(Class<?> defClass, String name, Class<?> type, byte refKind) {
    @Positive
    }

    @Positive
    public MemberName(Class<?> defClass, String name, MethodType type, byte refKind) {
    @Positive
    }

    @Positive
    public MemberName(byte refKind, Class<?> defClass, String name, Object type) {
    @Positive
    }

    @Positive
    public boolean hasReceiverTypeDispatch();

    @Positive
    public boolean isResolved();

    @Positive
    void initResolved(boolean isResolved);

    @Positive
    void checkForTypeAlias(Class<?> refc);

    @Positive
    @SuppressWarnings("LocalVariableHidesMemberVariable")
    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public IllegalAccessException makeAccessException(String message, Object from);

    @Positive
    public ReflectiveOperationException makeAccessException();

    @Positive
    static Factory getFactory();

    @Positive
    static class Factory {

    @Positive
        List<MemberName> getMembers(Class<?> defc, String matchName, Object matchType, int matchFlags, Class<?> lookupClass);

    @Positive
        public <NoSuchMemberException extends ReflectiveOperationException> MemberName resolveOrFail(byte refKind, MemberName m, Class<?> lookupClass, int allowedModes, Class<NoSuchMemberException> nsmClass) throws IllegalAccessException, NoSuchMemberException;

    @Positive
        public MemberName resolveOrNull(byte refKind, MemberName m, Class<?> lookupClass, int allowedModes);

    @Positive
        public List<MemberName> getMethods(Class<?> defc, boolean searchSupers, Class<?> lookupClass);

    @Positive
        public List<MemberName> getMethods(Class<?> defc, boolean searchSupers, String name, MethodType type, Class<?> lookupClass);

    @Positive
        public List<MemberName> getConstructors(Class<?> defc, Class<?> lookupClass);

    @Positive
        public List<MemberName> getFields(Class<?> defc, boolean searchSupers, Class<?> lookupClass);

    @Positive
        public List<MemberName> getFields(Class<?> defc, boolean searchSupers, String name, Class<?> type, Class<?> lookupClass);

    @Positive
        public List<MemberName> getNestedTypes(Class<?> defc, boolean searchSupers, Class<?> lookupClass);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
