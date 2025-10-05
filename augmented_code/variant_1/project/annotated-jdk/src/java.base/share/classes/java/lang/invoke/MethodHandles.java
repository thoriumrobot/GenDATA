/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.org.objectweb.asm.ClassReader;
    @Positive
import jdk.internal.org.objectweb.asm.Opcodes;
    @Positive
import jdk.internal.org.objectweb.asm.Type;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import sun.invoke.util.ValueConversions;
    @Positive
import sun.invoke.util.VerifyAccess;
    @Positive
import sun.invoke.util.Wrapper;
    @Positive
import sun.reflect.misc.ReflectUtil;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import java.lang.constant.ConstantDescs;
    @Positive
import java.lang.invoke.LambdaForm.BasicType;
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
import java.lang.reflect.ReflectPermission;
    @Positive
import java.nio.ByteOrder;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.BitSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.stream.Stream;
    @Positive
import static java.lang.invoke.LambdaForm.BasicType.V_TYPE;
    @Positive
import static java.lang.invoke.MethodHandleImpl.Intrinsic;
    @Positive
import static java.lang.invoke.MethodHandleNatives.Constants.*;
    @Positive
import static java.lang.invoke.MethodHandleStatics.newIllegalArgumentException;
    @Positive
import static java.lang.invoke.MethodType.methodType;

    @Positive
public class MethodHandles {

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    public static Lookup lookup();

    @Positive
    public static Lookup publicLookup();

    @Positive
    public static Lookup privateLookupIn(Class<?> targetClass, Lookup caller) throws IllegalAccessException;

    @Positive
    public static <T> T classData(Lookup caller, String name, Class<T> type) throws IllegalAccessException;

    @Positive
    public static <T> T classDataAt(Lookup caller, String name, Class<T> type, int index) throws IllegalAccessException;

    @Positive
    public static <T extends Member> T reflectAs(Class<T> expected, MethodHandle target);

    @Positive
    public static final class Lookup {

    @Positive
        public static final int PUBLIC;

    @Positive
        public static final int PRIVATE;

    @Positive
        public static final int PROTECTED;

    @Positive
        public static final int PACKAGE;

    @Positive
        public static final int MODULE;

    @Positive
        public static final int UNCONDITIONAL;

    @Positive
        public static final int ORIGINAL;

    @Positive
        public Class<?> lookupClass();

    @Positive
        public Class<?> previousLookupClass();

    @Positive
        public int lookupModes();

    @Positive
        public Lookup in(Class<?> requestedLookupClass);

    @Positive
        public Lookup dropLookupMode(int modeToDrop);

    @Positive
        public Class<?> defineClass(byte[] bytes) throws IllegalAccessException;

    @Positive
        public enum ClassOption {

    @Positive
            NESTMATE(NESTMATE_CLASS), STRONG(STRONG_LOADER_LINK);

    @Positive
            static int optionsToFlag(Set<ClassOption> options);
    @Positive
        }

    @Positive
        public Lookup defineHiddenClass(byte[] bytes, boolean initialize, ClassOption... options) throws IllegalAccessException;

    @Positive
        public Lookup defineHiddenClassWithClassData(byte[] bytes, Object classData, boolean initialize, ClassOption... options) throws IllegalAccessException;

    @Positive
        static class ClassFile {

    @Positive
            static ClassFile newInstanceNoCheck(String name, byte[] bytes);

    @Positive
            static ClassFile newInstance(byte[] bytes, String pkgName);
    @Positive
        }

    @Positive
        ClassDefiner makeHiddenClassDefiner(byte[] bytes);

    @Positive
        ClassDefiner makeHiddenClassDefiner(byte[] bytes, Set<ClassOption> options, boolean accessVmAnnotations);

    @Positive
        ClassDefiner makeHiddenClassDefiner(String name, byte[] bytes);

    @Positive
        static class ClassDefiner {

    @Positive
            String className();

    @Positive
            Class<?> defineClass(boolean initialize);

    @Positive
            Lookup defineClassAsLookup(boolean initialize);

    @Positive
            Class<?> defineClass(boolean initialize, Object classData);

    @Positive
            Lookup defineClassAsLookup(boolean initialize, Object classData);
    @Positive
        }

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        public MethodHandle findStatic(Class<?> refc, String name, MethodType type) throws NoSuchMethodException, IllegalAccessException;

    @Positive
        public MethodHandle findVirtual(Class<?> refc, String name, MethodType type) throws NoSuchMethodException, IllegalAccessException;

    @Positive
        public MethodHandle findConstructor(Class<?> refc, MethodType type) throws NoSuchMethodException, IllegalAccessException;

    @Positive
        public Class<?> findClass(String targetName) throws ClassNotFoundException, IllegalAccessException;

    @Positive
        public Class<?> ensureInitialized(Class<?> targetClass) throws IllegalAccessException;

    @Positive
        public Class<?> accessClass(Class<?> targetClass) throws IllegalAccessException;

    @Positive
        public MethodHandle findSpecial(Class<?> refc, String name, MethodType type, Class<?> specialCaller) throws NoSuchMethodException, IllegalAccessException;

    @Positive
        public MethodHandle findGetter(Class<?> refc, String name, Class<?> type) throws NoSuchFieldException, IllegalAccessException;

    @Positive
        public MethodHandle findSetter(Class<?> refc, String name, Class<?> type) throws NoSuchFieldException, IllegalAccessException;

    @Positive
        public VarHandle findVarHandle(Class<?> recv, String name, Class<?> type) throws NoSuchFieldException, IllegalAccessException;

    @Positive
        public MethodHandle findStaticGetter(Class<?> refc, String name, Class<?> type) throws NoSuchFieldException, IllegalAccessException;

    @Positive
        public MethodHandle findStaticSetter(Class<?> refc, String name, Class<?> type) throws NoSuchFieldException, IllegalAccessException;

    @Positive
        public VarHandle findStaticVarHandle(Class<?> decl, String name, Class<?> type) throws NoSuchFieldException, IllegalAccessException;

    @Positive
        public MethodHandle bind(Object receiver, String name, MethodType type) throws NoSuchMethodException, IllegalAccessException;

    @Positive
        public MethodHandle unreflect(Method m) throws IllegalAccessException;

    @Positive
        public MethodHandle unreflectSpecial(Method m, Class<?> specialCaller) throws IllegalAccessException;

    @Positive
        public MethodHandle unreflectConstructor(Constructor<?> c) throws IllegalAccessException;

    @Positive
        public MethodHandle unreflectGetter(Field f) throws IllegalAccessException;

    @Positive
        public MethodHandle unreflectSetter(Field f) throws IllegalAccessException;

    @Positive
        public VarHandle unreflectVarHandle(Field f) throws IllegalAccessException;

    @Positive
        public MethodHandleInfo revealDirect(MethodHandle target);

    @Positive
        MemberName resolveOrFail(byte refKind, Class<?> refc, String name, Class<?> type) throws NoSuchFieldException, IllegalAccessException;

    @Positive
        MemberName resolveOrFail(byte refKind, Class<?> refc, String name, MethodType type) throws NoSuchMethodException, IllegalAccessException;

    @Positive
        MemberName resolveOrFail(byte refKind, MemberName member) throws ReflectiveOperationException;

    @Positive
        MemberName resolveOrNull(byte refKind, MemberName member);

    @Positive
        MemberName resolveOrNull(byte refKind, Class<?> refc, String name, MethodType type);

    @Positive
        void checkSymbolicClass(Class<?> refc) throws IllegalAccessException;

    @Positive
        boolean isClassAccessible(Class<?> refc);

    @Positive
        void checkMethodName(byte refKind, String name) throws NoSuchMethodException;

    @Positive
        Lookup findBoundCallerLookup(MemberName m) throws IllegalAccessException;

    @Positive
        @Deprecated()
    @Positive
        public boolean hasPrivateAccess();

    @Positive
        public boolean hasFullPrivilegeAccess();

    @Positive
        void checkSecurityManager(Class<?> refc);

    @Positive
        void checkSecurityManager(Class<?> refc, MemberName m);

    @Positive
        void checkMethod(byte refKind, Class<?> refc, MemberName m) throws IllegalAccessException;

    @Positive
        void checkField(byte refKind, Class<?> refc, MemberName m) throws IllegalAccessException;

    @Positive
        void checkAccess(byte refKind, Class<?> refc, MemberName m) throws IllegalAccessException;

    @Positive
        String accessFailedMessage(Class<?> refc, MemberName m);

    @Positive
        MethodHandle linkMethodHandleConstant(byte refKind, Class<?> defc, String name, Object type) throws ReflectiveOperationException;
    @Positive
    }

    @Positive
    public static MethodHandle arrayConstructor(Class<?> arrayClass) throws IllegalArgumentException;

    @Positive
    public static MethodHandle arrayLength(Class<?> arrayClass) throws IllegalArgumentException;

    @Positive
    public static MethodHandle arrayElementGetter(Class<?> arrayClass) throws IllegalArgumentException;

    @Positive
    public static MethodHandle arrayElementSetter(Class<?> arrayClass) throws IllegalArgumentException;

    @Positive
    public static VarHandle arrayElementVarHandle(Class<?> arrayClass) throws IllegalArgumentException;

    @Positive
    public static VarHandle byteArrayViewVarHandle(Class<?> viewArrayClass, ByteOrder byteOrder) throws IllegalArgumentException;

    @Positive
    public static VarHandle byteBufferViewVarHandle(Class<?> viewArrayClass, ByteOrder byteOrder) throws IllegalArgumentException;

    @Positive
    public static MethodHandle spreadInvoker(MethodType type, int leadingArgCount);

    @Positive
    public static MethodHandle exactInvoker(MethodType type);

    @Positive
    public static MethodHandle invoker(MethodType type);

    @Positive
    public static MethodHandle varHandleExactInvoker(VarHandle.AccessMode accessMode, MethodType type);

    @Positive
    public static MethodHandle varHandleInvoker(VarHandle.AccessMode accessMode, MethodType type);

    @Positive
    static MethodHandle basicInvoker(MethodType type);

    @Positive
    public static MethodHandle explicitCastArguments(MethodHandle target, MethodType newType);

    @Positive
    public static MethodHandle permuteArguments(MethodHandle target, MethodType newType, int... reorder);

    @Positive
    static boolean permuteArgumentChecks(int[] reorder, MethodType newType, MethodType oldType);

    @Positive
    public static MethodHandle constant(Class<?> type, Object value);

    @Positive
    public static MethodHandle identity(Class<?> type);

    @Positive
    public static MethodHandle zero(Class<?> type);

    @Positive
    public static MethodHandle empty(MethodType type);

    @Positive
    public static MethodHandle insertArguments(MethodHandle target, int pos, Object... values);

    @Positive
    public static MethodHandle dropArguments(MethodHandle target, int pos, List<Class<?>> valueTypes);

    @Positive
    public static MethodHandle dropArguments(MethodHandle target, int pos, Class<?>... valueTypes);

    @Positive
    public static MethodHandle dropArgumentsToMatch(MethodHandle target, int skip, List<Class<?>> newTypes, int pos);

    @Positive
    public static MethodHandle dropReturn(MethodHandle target);

    @Positive
    public static MethodHandle filterArguments(MethodHandle target, int pos, MethodHandle... filters);

    @Positive
    static MethodHandle filterArgument(MethodHandle target, int pos, MethodHandle filter);

    @Positive
    public static MethodHandle collectArguments(MethodHandle target, int pos, MethodHandle filter);

    @Positive
    public static MethodHandle filterReturnValue(MethodHandle target, MethodHandle filter);

    @Positive
    static MethodHandle collectReturnValue(MethodHandle target, MethodHandle filter);

    @Positive
    public static MethodHandle foldArguments(MethodHandle target, MethodHandle combiner);

    @Positive
    public static MethodHandle foldArguments(MethodHandle target, int pos, MethodHandle combiner);

    @Positive
    static MethodHandle filterArgumentsWithCombiner(MethodHandle target, int position, MethodHandle combiner, int... argPositions);

    @Positive
    static MethodHandle foldArgumentsWithCombiner(MethodHandle target, int position, MethodHandle combiner, int... argPositions);

    @Positive
    public static MethodHandle guardWithTest(MethodHandle test, MethodHandle target, MethodHandle fallback);

    @Positive
    static <T> RuntimeException misMatchedTypes(String what, T t1, T t2);

    @Positive
    public static MethodHandle catchException(MethodHandle target, Class<? extends Throwable> exType, MethodHandle handler);

    @Positive
    public static MethodHandle throwException(Class<?> returnType, Class<? extends Throwable> exType);

    @Positive
    public static MethodHandle loop(MethodHandle[]... clauses);

    @Positive
    public static MethodHandle whileLoop(MethodHandle init, MethodHandle pred, MethodHandle body);

    @Positive
    public static MethodHandle doWhileLoop(MethodHandle init, MethodHandle body, MethodHandle pred);

    @Positive
    public static MethodHandle countedLoop(MethodHandle iterations, MethodHandle init, MethodHandle body);

    @Positive
    public static MethodHandle countedLoop(MethodHandle start, MethodHandle end, MethodHandle init, MethodHandle body);

    @Positive
    public static MethodHandle iteratedLoop(MethodHandle iterator, MethodHandle init, MethodHandle body);

    @Positive
    static MethodHandle swapArguments(MethodHandle mh, int i, int j);

    @Positive
    public static MethodHandle tryFinally(MethodHandle target, MethodHandle cleanup);

    @Positive
    public static MethodHandle tableSwitch(MethodHandle fallback, MethodHandle... targets);
    @Positive
}
