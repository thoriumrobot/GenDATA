/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2011, 2020, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
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
import jdk.internal.perf.PerfCounter;
    @Positive
import jdk.internal.vm.annotation.DontInline;
    @Positive
import jdk.internal.vm.annotation.Hidden;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.invoke.util.Wrapper;
    @Positive
import java.lang.annotation.ElementType;
    @Positive
import java.lang.annotation.Retention;
    @Positive
import java.lang.annotation.RetentionPolicy;
    @Positive
import java.lang.annotation.Target;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.HashMap;
    @Positive
import static java.lang.invoke.LambdaForm.BasicType.*;
    @Positive
import static java.lang.invoke.MethodHandleNatives.Constants.*;
    @Positive
import static java.lang.invoke.MethodHandleStatics.*;

    @Positive
class LambdaForm {

    @Positive
    public static final int VOID_RESULT, LAST_RESULT;

    @Positive
    static boolean debugNames();

    @Positive
    static void associateWithDebugName(LambdaForm form, String name);

    @Positive
    String lambdaName();

    @Positive
    LambdaForm customize(MethodHandle mh);

    @Positive
    LambdaForm uncustomize();

    @Positive
    boolean nameRefsAreLegal();

    @Positive
    BasicType returnType();

    @Positive
    BasicType parameterType(int n);

    @Positive
    Name parameter(int n);

    @Positive
    Object parameterConstraint(int n);

    @Positive
    int arity();

    @Positive
    int expressionCount();

    @Positive
    MethodType methodType();

    @Positive
    final String basicTypeSignature();

    @Positive
    static int signatureArity(String sig);

    @Positive
    static boolean isValidSignature(String sig);

    @Positive
    boolean isSelectAlternative(int pos);

    @Positive
    boolean isGuardWithCatch(int pos);

    @Positive
    boolean isTryFinally(int pos);

    @Positive
    boolean isTableSwitch(int pos);

    @Positive
    boolean isLoop(int pos);

    @Positive
    public void prepare();

    @Positive
    void compileToBytecode();

    @Positive
    @Hidden
    @Positive
    @DontInline
    @Positive
    Object interpretWithArguments(Object... argumentValues) throws Throwable;

    @Positive
    @Hidden
    @Positive
    @DontInline
    @Positive
    Object interpretName(Name name, Object[] values) throws Throwable;

    @Positive
    Object interpretWithArgumentsTracing(Object... argumentValues) throws Throwable;

    @Positive
    static void traceInterpreter(String event, Object obj, Object... args);

    @Positive
    static void traceInterpreter(String event, Object obj);

    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public boolean equals(LambdaForm that);

    @Positive
    public int hashCode();

    @Positive
    LambdaFormEditor editor();

    @Positive
    @Pure
    @Positive
    boolean contains(Name name);

    @Positive
    static class NamedFunction {

    @Positive
        MethodHandle resolvedHandle();

    @Positive
        synchronized void resolve();

    @Positive
        @Override
    @Positive
        public boolean equals(Object other);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Hidden
    @Positive
        Object invokeWithArguments(Object... arguments) throws Throwable;

    @Positive
        @Hidden
    @Positive
        Object invokeWithArgumentsTracing(Object[] arguments) throws Throwable;

    @Positive
        MethodType methodType();

    @Positive
        MemberName member();

    @Positive
        Class<?> memberDeclaringClassOrNull();

    @Positive
        BasicType returnType();

    @Positive
        BasicType parameterType(int n);

    @Positive
        int arity();

    @Positive
        public String toString();

    @Positive
        public boolean isIdentity();

    @Positive
        public boolean isConstantZero();

    @Positive
        public MethodHandleImpl.Intrinsic intrinsicName();

    @Positive
        public Object intrinsicData();
    @Positive
    }

    @Positive
    public static String basicTypeSignature(MethodType type);

    @Positive
    public static String shortenSignature(String signature);

    @Positive
    static final class Name {

    @Positive
        BasicType type();

    @Positive
        int index();

    @Positive
        boolean initIndex(int i);

    @Positive
        char typeChar();

    @Positive
        Name newIndex(int i);

    @Positive
        Name cloneWithIndex(int i);

    @Positive
        Name withConstraint(Object constraint);

    @Positive
        Name replaceName(Name oldName, Name newName);

    @Positive
        Name replaceNames(Name[] oldNames, Name[] newNames, int start, int end);

    @Positive
        void internArguments();

    @Positive
        boolean isParam();

    @Positive
        boolean isConstantZero();

    @Positive
        boolean refersTo(Class<?> declaringClass, String methodName);

    @Positive
        boolean isInvokeBasic();

    @Positive
        boolean isLinkerMethodInvoke();

    @Positive
        public String toString();

    @Positive
        public String debugString();

    @Positive
        public String paramString();

    @Positive
        public String exprString();

    @Positive
        int lastUseIndex(Name n);

    @Positive
        int useCount(Name n);

    @Positive
        public boolean equals(Name that);

    @Positive
        @Override
    @Positive
        public boolean equals(Object x);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    int lastUseIndex(Name n);

    @Positive
    int useCount(Name n);

    @Positive
    static Name argument(int which, BasicType type);

    @Positive
    static Name internArgument(Name n);

    @Positive
    static Name[] arguments(int extra, MethodType types);

    @Positive
    static LambdaForm identityForm(BasicType type);

    @Positive
    static LambdaForm zeroForm(BasicType type);

    @Positive
    static NamedFunction identity(BasicType type);

    @Positive
    static NamedFunction constantZero(BasicType type);

    @Positive
    @Target(ElementType.METHOD)
    @Positive
    @Retention(RetentionPolicy.RUNTIME)
    @Positive
    @interface Compiled {
    @Positive
    }

    @Positive
    final class Holder {
    @Positive
    }
    @Positive
}
