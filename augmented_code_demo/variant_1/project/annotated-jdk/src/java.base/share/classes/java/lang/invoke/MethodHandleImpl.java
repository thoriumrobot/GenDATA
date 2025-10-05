/*
    @Positive
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.access.JavaLangInvokeAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.invoke.NativeEntryPoint;
    @Positive
import jdk.internal.org.objectweb.asm.ClassWriter;
    @Positive
import jdk.internal.org.objectweb.asm.MethodVisitor;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.Hidden;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.invoke.empty.Empty;
    @Positive
import sun.invoke.util.ValueConversions;
    @Positive
import sun.invoke.util.VerifyType;
    @Positive
import sun.invoke.util.Wrapper;
    @Positive
import java.lang.invoke.MethodHandles.Lookup;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.nio.ByteOrder;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Stream;
    @Positive
import static java.lang.invoke.LambdaForm.*;
    @Positive
import static java.lang.invoke.MethodHandleStatics.*;
    @Positive
import static java.lang.invoke.MethodHandles.Lookup.IMPL_LOOKUP;
    @Positive
import static jdk.internal.org.objectweb.asm.Opcodes.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
abstract class MethodHandleImpl {

    @Positive
    static MethodHandle makeArrayElementAccessor(Class<?> arrayClass, ArrayAccess access);

    @Positive
    static InternalError unmatchedArrayAccess(ArrayAccess a);

    @Positive
    static final class ArrayAccessor {

    @Positive
        static int getElementI(int[] a, int i);

    @Positive
        static long getElementJ(long[] a, int i);

    @Positive
        static float getElementF(float[] a, int i);

    @Positive
        static double getElementD(double[] a, int i);

    @Positive
        static boolean getElementZ(boolean[] a, int i);

    @Positive
        static byte getElementB(byte[] a, int i);

    @Positive
        static short getElementS(short[] a, int i);

    @Positive
        static char getElementC(char[] a, int i);

    @Positive
        static Object getElementL(Object[] a, int i);

    @Positive
        static void setElementI(int[] a, int i, int x);

    @Positive
        static void setElementJ(long[] a, int i, long x);

    @Positive
        static void setElementF(float[] a, int i, float x);

    @Positive
        static void setElementD(double[] a, int i, double x);

    @Positive
        static void setElementZ(boolean[] a, int i, boolean x);

    @Positive
        static void setElementB(byte[] a, int i, byte x);

    @Positive
        static void setElementS(short[] a, int i, short x);

    @Positive
        static void setElementC(char[] a, int i, char x);

    @Positive
        static void setElementL(Object[] a, int i, Object x);

    @Positive
        static int lengthI(int[] a);

    @Positive
        static int lengthJ(long[] a);

    @Positive
        static int lengthF(float[] a);

    @Positive
        static int lengthD(double[] a);

    @Positive
        static int lengthZ(boolean[] a);

    @Positive
        static int lengthB(byte[] a);

    @Positive
        static int lengthS(short[] a);

    @Positive
        static int lengthC(char[] a);

    @Positive
        static int lengthL(Object[] a);

    @Positive
        static String name(Class<?> arrayClass, ArrayAccess access);

    @Positive
        static MethodType type(Class<?> arrayClass, ArrayAccess access);

    @Positive
        static MethodType correctType(Class<?> arrayClass, ArrayAccess access);

    @Positive
        static MethodHandle getAccessor(Class<?> arrayClass, ArrayAccess access);
    @Positive
    }

    @Positive
    static MethodHandle makePairwiseConvert(MethodHandle target, MethodType srcType, boolean strict, boolean monobox);

    @Positive
    static MethodHandle makePairwiseConvertByEditor(MethodHandle target, MethodType srcType, boolean strict, boolean monobox);

    @Positive
    static Object[] computeValueConversions(MethodType srcType, MethodType dstType, boolean strict, boolean monobox);

    @Positive
    static MethodHandle makePairwiseConvert(MethodHandle target, MethodType srcType, boolean strict);

    @Positive
    static Object valueConversion(Class<?> src, Class<?> dst, boolean strict, boolean monobox);

    @Positive
    static MethodHandle makeVarargsCollector(MethodHandle target, Class<?> arrayType);

    @Positive
    private static final class AsVarargsCollector extends DelegatingMethodHandle {

    @Positive
        @Override
    @Positive
        public boolean isVarargsCollector();

    @Positive
        @Override
    @Positive
        protected MethodHandle getTarget();

    @Positive
        @Override
    @Positive
        public MethodHandle asFixedArity();

    @Positive
        @Override
    @Positive
        MethodHandle setVarargs(MemberName member);

    @Positive
        @Override
    @Positive
        public MethodHandle withVarargs(boolean makeVarargs);

    @Positive
        @Override
    @Positive
        public MethodHandle asTypeUncached(MethodType newType);

    @Positive
        @Override
    @Positive
        boolean viewAsTypeChecks(MethodType newType, boolean strict);

    @Positive
        @Override
    @Positive
        public Object invokeWithArguments(Object... arguments) throws Throwable;
    @Positive
    }

    @Positive
    static void checkSpreadArgument(Object av, int n);

    @Positive
    @Hidden
    @Positive
    static MethodHandle selectAlternative(boolean testResult, MethodHandle target, MethodHandle fallback);

    @Positive
    @Hidden
    @Positive
    @jdk.internal.vm.annotation.IntrinsicCandidate
    @Positive
    static boolean profileBoolean(boolean result, int[] counters);

    @Positive
    @Hidden
    @Positive
    @jdk.internal.vm.annotation.IntrinsicCandidate
    @Positive
    static boolean isCompileConstant(Object obj);

    @Positive
    static MethodHandle makeGuardWithTest(MethodHandle test, MethodHandle target, MethodHandle fallback);

    @Positive
    static MethodHandle profile(MethodHandle target);

    @Positive
    static MethodHandle makeBlockInliningWrapper(MethodHandle target);

    @Positive
    private static final class Makers {
    @Positive
    }

    @Positive
    static class CountingWrapper extends DelegatingMethodHandle {

    @Positive
        @Hidden
    @Positive
        @Override
    @Positive
        protected MethodHandle getTarget();

    @Positive
        @Override
    @Positive
        public MethodHandle asTypeUncached(MethodType newType);

    @Positive
        boolean countDown();

    @Positive
        @Hidden
    @Positive
        static void maybeStopCounting(Object o1);
    @Positive
    }

    @Positive
    static LambdaForm makeGuardWithTestForm(MethodType basicType);

    @Positive
    static MethodHandle makeGuardWithCatch(MethodHandle target, Class<? extends Throwable> exType, MethodHandle catcher);

    @Positive
    @Hidden
    @Positive
    static Object guardWithCatch(MethodHandle target, Class<? extends Throwable> exType, MethodHandle catcher, Object... av) throws Throwable;

    @Positive
    static MethodHandle throwException(MethodType type);

    @Positive
    static <T extends Throwable> Empty throwException(T t) throws T;

    @Positive
    static MethodHandle fakeMethodHandleInvoke(MemberName method);

    @Positive
    static MethodHandle fakeVarHandleInvoke(MemberName method);

    @Positive
    static MethodHandle bindCaller(MethodHandle mh, Class<?> hostClass);

    @Positive
    private static class BindCaller {

    @Positive
        static MethodHandle bindCaller(MethodHandle mh, Class<?> hostClass);
    @Positive
    }

    @Positive
    private static final class WrappedMember extends DelegatingMethodHandle {

    @Positive
        @Override
    @Positive
        MemberName internalMemberName();

    @Positive
        @Override
    @Positive
        Class<?> internalCallerClass();

    @Positive
        @Override
    @Positive
        boolean isInvokeSpecial();

    @Positive
        @Override
    @Positive
        protected MethodHandle getTarget();

    @Positive
        @Override
    @Positive
        public MethodHandle asTypeUncached(MethodType newType);
    @Positive
    }

    @Positive
    static MethodHandle makeWrappedMember(MethodHandle target, MemberName member, boolean isInvokeSpecial);

    @Positive
    static final class IntrinsicMethodHandle extends DelegatingMethodHandle {

    @Positive
        @Override
    @Positive
        protected MethodHandle getTarget();

    @Positive
        @Override
    @Positive
        Intrinsic intrinsicName();

    @Positive
        @Override
    @Positive
        Object intrinsicData();

    @Positive
        @Override
    @Positive
        public MethodHandle asTypeUncached(MethodType newType);

    @Positive
        @Override
    @Positive
        String internalProperties();

    @Positive
        @Override
    @Positive
        public MethodHandle asCollector(Class<?> arrayType, int arrayLength);
    @Positive
    }

    @Positive
    static MethodHandle makeIntrinsic(MethodHandle target, Intrinsic intrinsicName);

    @Positive
    static MethodHandle makeIntrinsic(MethodHandle target, Intrinsic intrinsicName, Object intrinsicData);

    @Positive
    static MethodHandle makeIntrinsic(MethodType type, LambdaForm form, Intrinsic intrinsicName);

    @Positive
    static MethodHandle varargsArray(int nargs);

    @Positive
    static MethodHandle varargsArray(Class<?> arrayType, int nargs);

    @Positive
    static void assertSame(Object mh1, Object mh2);

    @Positive
    static NamedFunction getFunction(byte func);

    @Positive
    static MethodHandle makeLoop(Class<?> tloop, List<Class<?>> targs, List<MethodHandle> init, List<MethodHandle> step, List<MethodHandle> pred, List<MethodHandle> fini);

    @Positive
    static class LoopClauses {

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    @Hidden
    @Positive
    static Object loop(BasicType[] localTypes, LoopClauses clauseData, Object... av) throws Throwable;

    @Positive
    static boolean countedLoopPredicate(int limit, int counter);

    @Positive
    static int countedLoopStep(int limit, int counter);

    @Positive
    static Iterator<?> initIterator(Iterable<?> it);

    @Positive
    static boolean iteratePredicate(Iterator<?> it);

    @Positive
    static Object iterateNext(Iterator<?> it);

    @Positive
    static MethodHandle makeTryFinally(MethodHandle target, MethodHandle cleanup, Class<?> rtype, List<Class<?>> argTypes);

    @Positive
    @Hidden
    @Positive
    static Object tryFinally(MethodHandle target, MethodHandle cleanup, Object... av) throws Throwable;

    @Positive
    static class CasesHolder {

    @Positive
        public CasesHolder(MethodHandle[] cases) {
    @Positive
        }
    @Positive
    }

    @Positive
    static MethodHandle makeTableSwitch(MethodType type, MethodHandle defaultCase, MethodHandle[] caseActions);

    @Positive
    private static class TableSwitchCacheKey {

    @Positive
        public TableSwitchCacheKey(MethodType basicType, int numberOfCases) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    @Hidden
    @Positive
    static Object tableSwitch(int input, MethodHandle defaultCase, CasesHolder holder, Object[] args) throws Throwable;

    @Positive
    static MethodHandle getConstantHandle(int idx);
    @Positive
}

// CFWR semantic augmentation - variant 1
