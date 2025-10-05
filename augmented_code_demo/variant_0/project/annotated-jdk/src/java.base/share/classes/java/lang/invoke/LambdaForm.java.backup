/*
 * Copyright (c) 2011, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import jdk.internal.perf.PerfCounter;
import jdk.internal.vm.annotation.DontInline;
import jdk.internal.vm.annotation.Hidden;
import jdk.internal.vm.annotation.Stable;
import sun.invoke.util.Wrapper;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import static java.lang.invoke.LambdaForm.BasicType.*;
import static java.lang.invoke.MethodHandleNatives.Constants.*;
import static java.lang.invoke.MethodHandleStatics.*;

class LambdaForm {

    public static final int VOID_RESULT, LAST_RESULT;

    static boolean debugNames();

    static void associateWithDebugName(LambdaForm form, String name);

    String lambdaName();

    LambdaForm customize(MethodHandle mh);

    LambdaForm uncustomize();

    boolean nameRefsAreLegal();

    BasicType returnType();

    BasicType parameterType(int n);

    Name parameter(int n);

    Object parameterConstraint(int n);

    int arity();

    int expressionCount();

    MethodType methodType();

    final String basicTypeSignature();

    static int signatureArity(String sig);

    static boolean isValidSignature(String sig);

    boolean isSelectAlternative(int pos);

    boolean isGuardWithCatch(int pos);

    boolean isTryFinally(int pos);

    boolean isTableSwitch(int pos);

    boolean isLoop(int pos);

    public void prepare();

    void compileToBytecode();

    @Hidden
    @DontInline
    Object interpretWithArguments(Object... argumentValues) throws Throwable;

    @Hidden
    @DontInline
    Object interpretName(Name name, Object[] values) throws Throwable;

    Object interpretWithArgumentsTracing(Object... argumentValues) throws Throwable;

    static void traceInterpreter(String event, Object obj, Object... args);

    static void traceInterpreter(String event, Object obj);

    public String toString();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public boolean equals(LambdaForm that);

    public int hashCode();

    LambdaFormEditor editor();

    @Pure
    boolean contains(Name name);

    static class NamedFunction {

        MethodHandle resolvedHandle();

        synchronized void resolve();

        @Override
        public boolean equals(Object other);

        @Override
        public int hashCode();

        @Hidden
        Object invokeWithArguments(Object... arguments) throws Throwable;

        @Hidden
        Object invokeWithArgumentsTracing(Object[] arguments) throws Throwable;

        MethodType methodType();

        MemberName member();

        Class<?> memberDeclaringClassOrNull();

        BasicType returnType();

        BasicType parameterType(int n);

        int arity();

        public String toString();

        public boolean isIdentity();

        public boolean isConstantZero();

        public MethodHandleImpl.Intrinsic intrinsicName();

        public Object intrinsicData();
    }

    public static String basicTypeSignature(MethodType type);

    public static String shortenSignature(String signature);

    static final class Name {

        BasicType type();

        int index();

        boolean initIndex(int i);

        char typeChar();

        Name newIndex(int i);

        Name cloneWithIndex(int i);

        Name withConstraint(Object constraint);

        Name replaceName(Name oldName, Name newName);

        Name replaceNames(Name[] oldNames, Name[] newNames, int start, int end);

        void internArguments();

        boolean isParam();

        boolean isConstantZero();

        boolean refersTo(Class<?> declaringClass, String methodName);

        boolean isInvokeBasic();

        boolean isLinkerMethodInvoke();

        public String toString();

        public String debugString();

        public String paramString();

        public String exprString();

        int lastUseIndex(Name n);

        int useCount(Name n);

        public boolean equals(Name that);

        @Override
        public boolean equals(Object x);

        @Override
        public int hashCode();
    }

    int lastUseIndex(Name n);

    int useCount(Name n);

    static Name argument(int which, BasicType type);

    static Name internArgument(Name n);

    static Name[] arguments(int extra, MethodType types);

    static LambdaForm identityForm(BasicType type);

    static LambdaForm zeroForm(BasicType type);

    static NamedFunction identity(BasicType type);

    static NamedFunction constantZero(BasicType type);

    @Target(ElementType.METHOD)
    @Retention(RetentionPolicy.RUNTIME)
    @interface Compiled {
    }

    final class Holder {
    }
}
