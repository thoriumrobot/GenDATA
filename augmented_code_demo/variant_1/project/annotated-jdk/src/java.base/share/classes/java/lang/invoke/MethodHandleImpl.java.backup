/*
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.access.JavaLangInvokeAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.invoke.NativeEntryPoint;
import jdk.internal.org.objectweb.asm.ClassWriter;
import jdk.internal.org.objectweb.asm.MethodVisitor;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import jdk.internal.vm.annotation.ForceInline;
import jdk.internal.vm.annotation.Hidden;
import jdk.internal.vm.annotation.Stable;
import sun.invoke.empty.Empty;
import sun.invoke.util.ValueConversions;
import sun.invoke.util.VerifyType;
import sun.invoke.util.Wrapper;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.Array;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Stream;
import static java.lang.invoke.LambdaForm.*;
import static java.lang.invoke.MethodHandleStatics.*;
import static java.lang.invoke.MethodHandles.Lookup.IMPL_LOOKUP;
import static jdk.internal.org.objectweb.asm.Opcodes.*;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
abstract class MethodHandleImpl {

    static MethodHandle makeArrayElementAccessor(Class<?> arrayClass, ArrayAccess access);

    static InternalError unmatchedArrayAccess(ArrayAccess a);

    static final class ArrayAccessor {

        static int getElementI(int[] a, int i);

        static long getElementJ(long[] a, int i);

        static float getElementF(float[] a, int i);

        static double getElementD(double[] a, int i);

        static boolean getElementZ(boolean[] a, int i);

        static byte getElementB(byte[] a, int i);

        static short getElementS(short[] a, int i);

        static char getElementC(char[] a, int i);

        static Object getElementL(Object[] a, int i);

        static void setElementI(int[] a, int i, int x);

        static void setElementJ(long[] a, int i, long x);

        static void setElementF(float[] a, int i, float x);

        static void setElementD(double[] a, int i, double x);

        static void setElementZ(boolean[] a, int i, boolean x);

        static void setElementB(byte[] a, int i, byte x);

        static void setElementS(short[] a, int i, short x);

        static void setElementC(char[] a, int i, char x);

        static void setElementL(Object[] a, int i, Object x);

        static int lengthI(int[] a);

        static int lengthJ(long[] a);

        static int lengthF(float[] a);

        static int lengthD(double[] a);

        static int lengthZ(boolean[] a);

        static int lengthB(byte[] a);

        static int lengthS(short[] a);

        static int lengthC(char[] a);

        static int lengthL(Object[] a);

        static String name(Class<?> arrayClass, ArrayAccess access);

        static MethodType type(Class<?> arrayClass, ArrayAccess access);

        static MethodType correctType(Class<?> arrayClass, ArrayAccess access);

        static MethodHandle getAccessor(Class<?> arrayClass, ArrayAccess access);
    }

    static MethodHandle makePairwiseConvert(MethodHandle target, MethodType srcType, boolean strict, boolean monobox);

    static MethodHandle makePairwiseConvertByEditor(MethodHandle target, MethodType srcType, boolean strict, boolean monobox);

    static Object[] computeValueConversions(MethodType srcType, MethodType dstType, boolean strict, boolean monobox);

    static MethodHandle makePairwiseConvert(MethodHandle target, MethodType srcType, boolean strict);

    static Object valueConversion(Class<?> src, Class<?> dst, boolean strict, boolean monobox);

    static MethodHandle makeVarargsCollector(MethodHandle target, Class<?> arrayType);

    private static final class AsVarargsCollector extends DelegatingMethodHandle {

        @Override
        public boolean isVarargsCollector();

        @Override
        protected MethodHandle getTarget();

        @Override
        public MethodHandle asFixedArity();

        @Override
        MethodHandle setVarargs(MemberName member);

        @Override
        public MethodHandle withVarargs(boolean makeVarargs);

        @Override
        public MethodHandle asTypeUncached(MethodType newType);

        @Override
        boolean viewAsTypeChecks(MethodType newType, boolean strict);

        @Override
        public Object invokeWithArguments(Object... arguments) throws Throwable;
    }

    static void checkSpreadArgument(Object av, int n);

    @Hidden
    static MethodHandle selectAlternative(boolean testResult, MethodHandle target, MethodHandle fallback);

    @Hidden
    @jdk.internal.vm.annotation.IntrinsicCandidate
    static boolean profileBoolean(boolean result, int[] counters);

    @Hidden
    @jdk.internal.vm.annotation.IntrinsicCandidate
    static boolean isCompileConstant(Object obj);

    static MethodHandle makeGuardWithTest(MethodHandle test, MethodHandle target, MethodHandle fallback);

    static MethodHandle profile(MethodHandle target);

    static MethodHandle makeBlockInliningWrapper(MethodHandle target);

    private static final class Makers {
    }

    static class CountingWrapper extends DelegatingMethodHandle {

        @Hidden
        @Override
        protected MethodHandle getTarget();

        @Override
        public MethodHandle asTypeUncached(MethodType newType);

        boolean countDown();

        @Hidden
        static void maybeStopCounting(Object o1);
    }

    static LambdaForm makeGuardWithTestForm(MethodType basicType);

    static MethodHandle makeGuardWithCatch(MethodHandle target, Class<? extends Throwable> exType, MethodHandle catcher);

    @Hidden
    static Object guardWithCatch(MethodHandle target, Class<? extends Throwable> exType, MethodHandle catcher, Object... av) throws Throwable;

    static MethodHandle throwException(MethodType type);

    static <T extends Throwable> Empty throwException(T t) throws T;

    static MethodHandle fakeMethodHandleInvoke(MemberName method);

    static MethodHandle fakeVarHandleInvoke(MemberName method);

    static MethodHandle bindCaller(MethodHandle mh, Class<?> hostClass);

    private static class BindCaller {

        static MethodHandle bindCaller(MethodHandle mh, Class<?> hostClass);
    }

    private static final class WrappedMember extends DelegatingMethodHandle {

        @Override
        MemberName internalMemberName();

        @Override
        Class<?> internalCallerClass();

        @Override
        boolean isInvokeSpecial();

        @Override
        protected MethodHandle getTarget();

        @Override
        public MethodHandle asTypeUncached(MethodType newType);
    }

    static MethodHandle makeWrappedMember(MethodHandle target, MemberName member, boolean isInvokeSpecial);

    static final class IntrinsicMethodHandle extends DelegatingMethodHandle {

        @Override
        protected MethodHandle getTarget();

        @Override
        Intrinsic intrinsicName();

        @Override
        Object intrinsicData();

        @Override
        public MethodHandle asTypeUncached(MethodType newType);

        @Override
        String internalProperties();

        @Override
        public MethodHandle asCollector(Class<?> arrayType, int arrayLength);
    }

    static MethodHandle makeIntrinsic(MethodHandle target, Intrinsic intrinsicName);

    static MethodHandle makeIntrinsic(MethodHandle target, Intrinsic intrinsicName, Object intrinsicData);

    static MethodHandle makeIntrinsic(MethodType type, LambdaForm form, Intrinsic intrinsicName);

    static MethodHandle varargsArray(int nargs);

    static MethodHandle varargsArray(Class<?> arrayType, int nargs);

    static void assertSame(Object mh1, Object mh2);

    static NamedFunction getFunction(byte func);

    static MethodHandle makeLoop(Class<?> tloop, List<Class<?>> targs, List<MethodHandle> init, List<MethodHandle> step, List<MethodHandle> pred, List<MethodHandle> fini);

    static class LoopClauses {

        @Override
        public String toString();
    }

    @Hidden
    static Object loop(BasicType[] localTypes, LoopClauses clauseData, Object... av) throws Throwable;

    static boolean countedLoopPredicate(int limit, int counter);

    static int countedLoopStep(int limit, int counter);

    static Iterator<?> initIterator(Iterable<?> it);

    static boolean iteratePredicate(Iterator<?> it);

    static Object iterateNext(Iterator<?> it);

    static MethodHandle makeTryFinally(MethodHandle target, MethodHandle cleanup, Class<?> rtype, List<Class<?>> argTypes);

    @Hidden
    static Object tryFinally(MethodHandle target, MethodHandle cleanup, Object... av) throws Throwable;

    static class CasesHolder {

        public CasesHolder(MethodHandle[] cases) {
        }
    }

    static MethodHandle makeTableSwitch(MethodType type, MethodHandle defaultCase, MethodHandle[] caseActions);

    private static class TableSwitchCacheKey {

        public TableSwitchCacheKey(MethodType basicType, int numberOfCases) {
        }

        @Override
        public boolean equals(Object o);

        @Override
        public int hashCode();
    }

    @Hidden
    static Object tableSwitch(int input, MethodHandle defaultCase, CasesHolder holder, Object[] args) throws Throwable;

    static MethodHandle getConstantHandle(int idx);
}
