/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.invoke.MethodHandle;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.MethodType;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.lang.reflect.RecordComponent;
    @Positive
import java.lang.reflect.UndeclaredThrowableException;
    @Positive
import java.lang.reflect.Member;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.lang.reflect.Proxy;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.MessageDigest;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.PermissionCollection;
    @Positive
import java.security.Permissions;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.reflect.ReflectionFactory;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.access.JavaSecurityAccess;
    @Positive
import sun.reflect.misc.ReflectUtil;
    @Positive
import static java.io.ObjectStreamField.*;

    @Positive
@AnnotatedFor({ "index", "lock", "nullness", "signature" })
    @Positive
public class ObjectStreamClass implements Serializable {

    @Positive
    public static final ObjectStreamField[] NO_FIELDS;

    @Positive
    private static class Caches {
    @Positive
    }

    @Positive
    private static class ExceptionInfo {

    @Positive
        InvalidClassException newInvalidClassException();
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public static ObjectStreamClass lookup(Class<?> cl);

    @Positive
    public static ObjectStreamClass lookupAny(Class<?> cl);

    @Positive
    @BinaryName
    @Positive
    public String getName();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public long getSerialVersionUID();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public Class<?> forClass();

    @Positive
    public ObjectStreamField[] getFields();

    @Positive
    @Nullable
    @Positive
    public ObjectStreamField getField(String name);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied ObjectStreamClass this);

    @Positive
    static ObjectStreamClass lookup(Class<?> cl, boolean all);

    @Positive
    private static class EntryFuture {

    @Positive
        synchronized boolean set(Object entry);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        synchronized Object get();

    @Positive
        Thread getOwner();
    @Positive
    }

    @Positive
    void initProxy(Class<?> cl, ClassNotFoundException resolveEx, ObjectStreamClass superDesc) throws InvalidClassException;

    @Positive
    void initNonProxy(ObjectStreamClass model, Class<?> cl, ClassNotFoundException resolveEx, ObjectStreamClass superDesc) throws InvalidClassException;

    @Positive
    void readNonProxy(ObjectInputStream in) throws IOException, ClassNotFoundException;

    @Positive
    void writeNonProxy(ObjectOutputStream out) throws IOException;

    @Positive
    ClassNotFoundException getResolveException();

    @Positive
    final void checkInitialized() throws InvalidClassException;

    @Positive
    void checkDeserialize() throws InvalidClassException;

    @Positive
    void checkSerialize() throws InvalidClassException;

    @Positive
    void checkDefaultSerialize() throws InvalidClassException;

    @Positive
    ObjectStreamClass getSuperDesc();

    @Positive
    ObjectStreamClass getLocalDesc();

    @Positive
    ObjectStreamField[] getFields(boolean copy);

    @Positive
    ObjectStreamField getField(String name, Class<?> type);

    @Positive
    boolean isProxy();

    @Positive
    boolean isEnum();

    @Positive
    boolean isRecord();

    @Positive
    boolean isExternalizable();

    @Positive
    boolean isSerializable();

    @Positive
    boolean hasBlockExternalData();

    @Positive
    boolean hasWriteObjectData();

    @Positive
    boolean isInstantiable();

    @Positive
    boolean hasWriteObjectMethod();

    @Positive
    boolean hasReadObjectMethod();

    @Positive
    boolean hasReadObjectNoDataMethod();

    @Positive
    boolean hasWriteReplaceMethod();

    @Positive
    boolean hasReadResolveMethod();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    Object newInstance() throws InstantiationException, InvocationTargetException, UnsupportedOperationException;

    @Positive
    void invokeWriteObject(Object obj, ObjectOutputStream out) throws IOException, UnsupportedOperationException;

    @Positive
    void invokeReadObject(Object obj, ObjectInputStream in) throws ClassNotFoundException, IOException, UnsupportedOperationException;

    @Positive
    void invokeReadObjectNoData(Object obj) throws IOException, UnsupportedOperationException;

    @Positive
    Object invokeWriteReplace(Object obj) throws IOException, UnsupportedOperationException;

    @Positive
    Object invokeReadResolve(Object obj) throws IOException, UnsupportedOperationException;

    @Positive
    static class ClassDataSlot {
    @Positive
    }

    @Positive
    ClassDataSlot[] getClassDataLayout() throws InvalidClassException;

    @Positive
    int getPrimDataSize();

    @Positive
    int getNumObjFields();

    @Positive
    void getPrimFieldValues(Object obj, byte[] buf);

    @Positive
    void setPrimFieldValues(Object obj, byte[] buf);

    @Positive
    void getObjFieldValues(Object obj, Object[] vals);

    @Positive
    void checkObjFieldValueTypes(Object obj, Object[] vals);

    @Positive
    void setObjFieldValues(Object obj, Object[] vals);

    @Positive
    MethodHandle getRecordConstructor();

    @Positive
    private static class MemberSignature {

    @Positive
        public final Member member;

    @Positive
        public final String name;

    @Positive
        public final String signature;

    @Positive
        public MemberSignature(Field field) {
    @Positive
        }

    @Positive
        public MemberSignature(Constructor<?> cons) {
    @Positive
        }

    @Positive
        public MemberSignature(Method meth) {
    @Positive
        }
    @Positive
    }

    @Positive
    private static class FieldReflector {

    @Positive
        ObjectStreamField[] getFields();

    @Positive
        void getPrimFieldValues(Object obj, byte[] buf);

    @Positive
        void setPrimFieldValues(Object obj, byte[] buf);

    @Positive
        void getObjFieldValues(Object obj, Object[] vals);

    @Positive
        void checkObjectFieldValueTypes(Object obj, Object[] vals);

    @Positive
        void setObjFieldValues(Object obj, Object[] vals);
    @Positive
    }

    @Positive
    private static class FieldReflectorKey extends WeakReference<Class<?>> {

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    static void processQueue(ReferenceQueue<Class<?>> queue, ConcurrentMap<? extends WeakReference<Class<?>>, ?> map);

    @Positive
    static class WeakClassKey extends WeakReference<Class<?>> {

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    private static final class DeserializationConstructorsCache extends ConcurrentHashMap<DeserializationConstructorsCache.Key, MethodHandle> {

    @Positive
        MethodHandle get(ObjectStreamField[] fields);

    @Positive
        synchronized MethodHandle putIfAbsentAndGet(ObjectStreamField[] fields, MethodHandle mh);

    @Positive
        static abstract class Key {

    @Positive
            abstract int length();

    @Positive
            abstract String fieldName(int i);

    @Positive
            abstract Class<?> fieldType(int i);

    @Positive
            @Override
    @Positive
            public final int hashCode();

    @Positive
            @Override
    @Positive
            public final boolean equals(Object obj);

    @Positive
            static final class Lookup extends Key {

    @Positive
                @Override
    @Positive
                int length();

    @Positive
                @Override
    @Positive
                String fieldName(int i);

    @Positive
                @Override
    @Positive
                Class<?> fieldType(int i);
    @Positive
            }

    @Positive
            static final class Impl extends Key {

    @Positive
                @Override
    @Positive
                int length();

    @Positive
                @Override
    @Positive
                String fieldName(int i);

    @Positive
                @Override
    @Positive
                Class<?> fieldType(int i);
    @Positive
            }
    @Positive
        }
    @Positive
    }

    @Positive
    static final class RecordSupport {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        static MethodHandle deserializationCtr(ObjectStreamClass desc);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
