/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.UndeclaredThrowableException;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Proxy;
import java.security.AccessControlContext;
import java.security.AccessController;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.PermissionCollection;
import java.security.Permissions;
import java.security.PrivilegedAction;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import jdk.internal.misc.Unsafe;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import jdk.internal.reflect.ReflectionFactory;
import jdk.internal.access.SharedSecrets;
import jdk.internal.access.JavaSecurityAccess;
import sun.reflect.misc.ReflectUtil;
import static java.io.ObjectStreamField.*;

@AnnotatedFor({ "index", "lock", "nullness", "signature" })
public class ObjectStreamClass implements Serializable {

    public static final ObjectStreamField[] NO_FIELDS;

    private static class Caches {
    }

    private static class ExceptionInfo {

        InvalidClassException newInvalidClassException();
    }

    @Nullable
    public static ObjectStreamClass lookup(Class<?> cl);

    public static ObjectStreamClass lookupAny(Class<?> cl);

    @BinaryName
    public String getName();

    @SuppressWarnings("removal")
    public long getSerialVersionUID();

    @SuppressWarnings("removal")
    @CallerSensitive
    @Nullable
    public Class<?> forClass();

    public ObjectStreamField[] getFields();

    @Nullable
    public ObjectStreamField getField(String name);

    @SideEffectFree
    public String toString(@GuardSatisfied ObjectStreamClass this);

    static ObjectStreamClass lookup(Class<?> cl, boolean all);

    private static class EntryFuture {

        synchronized boolean set(Object entry);

        @SuppressWarnings("removal")
        synchronized Object get();

        Thread getOwner();
    }

    void initProxy(Class<?> cl, ClassNotFoundException resolveEx, ObjectStreamClass superDesc) throws InvalidClassException;

    void initNonProxy(ObjectStreamClass model, Class<?> cl, ClassNotFoundException resolveEx, ObjectStreamClass superDesc) throws InvalidClassException;

    void readNonProxy(ObjectInputStream in) throws IOException, ClassNotFoundException;

    void writeNonProxy(ObjectOutputStream out) throws IOException;

    ClassNotFoundException getResolveException();

    final void checkInitialized() throws InvalidClassException;

    void checkDeserialize() throws InvalidClassException;

    void checkSerialize() throws InvalidClassException;

    void checkDefaultSerialize() throws InvalidClassException;

    ObjectStreamClass getSuperDesc();

    ObjectStreamClass getLocalDesc();

    ObjectStreamField[] getFields(boolean copy);

    ObjectStreamField getField(String name, Class<?> type);

    boolean isProxy();

    boolean isEnum();

    boolean isRecord();

    boolean isExternalizable();

    boolean isSerializable();

    boolean hasBlockExternalData();

    boolean hasWriteObjectData();

    boolean isInstantiable();

    boolean hasWriteObjectMethod();

    boolean hasReadObjectMethod();

    boolean hasReadObjectNoDataMethod();

    boolean hasWriteReplaceMethod();

    boolean hasReadResolveMethod();

    @SuppressWarnings("removal")
    Object newInstance() throws InstantiationException, InvocationTargetException, UnsupportedOperationException;

    void invokeWriteObject(Object obj, ObjectOutputStream out) throws IOException, UnsupportedOperationException;

    void invokeReadObject(Object obj, ObjectInputStream in) throws ClassNotFoundException, IOException, UnsupportedOperationException;

    void invokeReadObjectNoData(Object obj) throws IOException, UnsupportedOperationException;

    Object invokeWriteReplace(Object obj) throws IOException, UnsupportedOperationException;

    Object invokeReadResolve(Object obj) throws IOException, UnsupportedOperationException;

    static class ClassDataSlot {
    }

    ClassDataSlot[] getClassDataLayout() throws InvalidClassException;

    int getPrimDataSize();

    int getNumObjFields();

    void getPrimFieldValues(Object obj, byte[] buf);

    void setPrimFieldValues(Object obj, byte[] buf);

    void getObjFieldValues(Object obj, Object[] vals);

    void checkObjFieldValueTypes(Object obj, Object[] vals);

    void setObjFieldValues(Object obj, Object[] vals);

    MethodHandle getRecordConstructor();

    private static class MemberSignature {

        public final Member member;

        public final String name;

        public final String signature;

        public MemberSignature(Field field) {
        }

        public MemberSignature(Constructor<?> cons) {
        }

        public MemberSignature(Method meth) {
        }
    }

    private static class FieldReflector {

        ObjectStreamField[] getFields();

        void getPrimFieldValues(Object obj, byte[] buf);

        void setPrimFieldValues(Object obj, byte[] buf);

        void getObjFieldValues(Object obj, Object[] vals);

        void checkObjectFieldValueTypes(Object obj, Object[] vals);

        void setObjFieldValues(Object obj, Object[] vals);
    }

    private static class FieldReflectorKey extends WeakReference<Class<?>> {

        public int hashCode();

        public boolean equals(Object obj);
    }

    static void processQueue(ReferenceQueue<Class<?>> queue, ConcurrentMap<? extends WeakReference<Class<?>>, ?> map);

    static class WeakClassKey extends WeakReference<Class<?>> {

        public int hashCode();

        public boolean equals(Object obj);
    }

    @SuppressWarnings("serial")
    private static final class DeserializationConstructorsCache extends ConcurrentHashMap<DeserializationConstructorsCache.Key, MethodHandle> {

        MethodHandle get(ObjectStreamField[] fields);

        synchronized MethodHandle putIfAbsentAndGet(ObjectStreamField[] fields, MethodHandle mh);

        static abstract class Key {

            abstract int length();

            abstract String fieldName(int i);

            abstract Class<?> fieldType(int i);

            @Override
            public final int hashCode();

            @Override
            public final boolean equals(Object obj);

            static final class Lookup extends Key {

                @Override
                int length();

                @Override
                String fieldName(int i);

                @Override
                Class<?> fieldType(int i);
            }

            static final class Impl extends Key {

                @Override
                int length();

                @Override
                String fieldName(int i);

                @Override
                Class<?> fieldType(int i);
            }
        }
    }

    static final class RecordSupport {

        @SuppressWarnings("removal")
        static MethodHandle deserializationCtr(ObjectStreamClass desc);
    }
}
