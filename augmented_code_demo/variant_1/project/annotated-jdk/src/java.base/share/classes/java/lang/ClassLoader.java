/*
    @Positive
 * Copyright (c) 2013, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * Copyright (c) 2019, Azul Systems, Inc. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.RequiresNonNull;
    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.checker.signature.qual.FullyQualifiedName;
    @Positive
import org.checkerframework.common.reflection.qual.ForName;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.io.File;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.net.URL;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.CodeSource;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import jdk.internal.loader.BootLoader;
    @Positive
import jdk.internal.loader.BuiltinClassLoader;
    @Positive
import jdk.internal.loader.ClassLoaders;
    @Positive
import jdk.internal.loader.NativeLibrary;
    @Positive
import jdk.internal.loader.NativeLibraries;
    @Positive
import jdk.internal.perf.PerfCounter;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.util.StaticProperty;
    @Positive
import sun.reflect.misc.ReflectUtil;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness", "signature" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class ClassLoader {

    @Positive
    private static class ParallelLoaders {

    @Positive
        static boolean register(Class<? extends ClassLoader> c);

    @Positive
        static boolean isRegistered(Class<? extends ClassLoader> c);
    @Positive
    }

    @Positive
    void addClass(Class<?> c);

    @Positive
    protected ClassLoader(String name, ClassLoader parent) {
    @Positive
    }

    @Positive
    protected ClassLoader(@Nullable ClassLoader parent) {
    @Positive
    }

    @Positive
    protected ClassLoader() {
    @Positive
    }

    @Positive
    public String getName();

    @Positive
    final String name();

    @Positive
    @ForName
    @Positive
    public Class<?> loadClass(@BinaryName String name) throws ClassNotFoundException;

    @Positive
    @ForName
    @Positive
    protected Class<?> loadClass(@BinaryName String name, boolean resolve) throws ClassNotFoundException;

    @Positive
    @ForName
    @Positive
    final Class<?> loadClass(Module module, @BinaryName String name);

    @Positive
    protected Object getClassLoadingLock(String className);

    @Positive
    protected Class<?> findClass(@BinaryName String name) throws ClassNotFoundException;

    @Positive
    protected Class<?> findClass(String moduleName, String name);

    @Positive
    @Deprecated()
    @Positive
    @SuppressWarnings("signature")
    @Positive
    protected final Class<?> defineClass(byte[] b, int off, int len) throws ClassFormatError;

    @Positive
    protected final Class<?> defineClass(@Nullable @BinaryName String name, byte[] b, int off, int len) throws ClassFormatError;

    @Positive
    protected final Class<?> defineClass(@Nullable @BinaryName String name, byte[] b, int off, int len, @Nullable ProtectionDomain protectionDomain) throws ClassFormatError;

    @Positive
    protected final Class<?> defineClass(@Nullable @BinaryName String name, java.nio.ByteBuffer b, @Nullable ProtectionDomain protectionDomain) throws ClassFormatError;

    @Positive
    static native Class<?> defineClass1(ClassLoader loader, @BinaryName String name, byte[] b, int off, int len, ProtectionDomain pd, String source);

    @Positive
    static native Class<?> defineClass2(ClassLoader loader, @BinaryName String name, java.nio.ByteBuffer b, int off, int len, ProtectionDomain pd, String source);

    @Positive
    static native Class<?> defineClass0(ClassLoader loader, Class<?> lookup, String name, byte[] b, int off, int len, ProtectionDomain pd, boolean initialize, int flags, Object classData);

    @Positive
    protected final void resolveClass(Class<?> c);

    @Positive
    protected final Class<?> findSystemClass(@BinaryName String name) throws ClassNotFoundException;

    @Positive
    @Nullable
    @Positive
    static Class<?> findBootstrapClassOrNull(String name);

    @Positive
    @Nullable
    @Positive
    protected final Class<?> findLoadedClass(@BinaryName String name);

    @Positive
    protected final void setSigners(Class<?> c, Object[] signers);

    @Positive
    protected URL findResource(String moduleName, String name) throws IOException;

    @Positive
    @Nullable
    @Positive
    public URL getResource(String name);

    @Positive
    public Enumeration<URL> getResources(String name) throws IOException;

    @Positive
    public Stream<URL> resources(String name);

    @Positive
    @Nullable
    @Positive
    protected URL findResource(String name);

    @Positive
    protected Enumeration<URL> findResources(String name) throws IOException;

    @Positive
    @CallerSensitive
    @Positive
    protected static boolean registerAsParallelCapable();

    @Positive
    public final boolean isRegisteredAsParallelCapable();

    @Positive
    @Nullable
    @Positive
    public static URL getSystemResource(String name);

    @Positive
    public static Enumeration<URL> getSystemResources(String name) throws IOException;

    @Positive
    @Nullable
    @Positive
    public InputStream getResourceAsStream(String name);

    @Positive
    @Nullable
    @Positive
    public static InputStream getSystemResourceAsStream(String name);

    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public final ClassLoader getParent();

    @Positive
    public final Module getUnnamedModule();

    @Positive
    @CallerSensitive
    @Positive
    public static ClassLoader getPlatformClassLoader();

    @Positive
    @CallerSensitive
    @Positive
    public static ClassLoader getSystemClassLoader();

    @Positive
    static ClassLoader getBuiltinPlatformClassLoader();

    @Positive
    static ClassLoader getBuiltinAppClassLoader();

    @Positive
    static synchronized ClassLoader initSystemClassLoader();

    @Positive
    boolean isAncestor(ClassLoader cl);

    @Positive
    @Nullable
    @Positive
    static ClassLoader getClassLoader(Class<?> caller);

    @Positive
    static void checkClassLoaderPermission(ClassLoader cl, Class<?> caller);

    @Positive
    Package definePackage(Class<?> c);

    @Positive
    Package definePackage(String name, Module m);

    @Positive
    protected Package definePackage(@FullyQualifiedName String name, @Nullable String specTitle, @Nullable String specVersion, @Nullable String specVendor, @Nullable String implTitle, @Nullable String implVersion, @Nullable String implVendor, @Nullable URL sealBase);

    @Positive
    public final Package getDefinedPackage(String name);

    @Positive
    public final Package[] getDefinedPackages();

    @Positive
    @Deprecated()
    @Positive
    @Nullable
    @Positive
    protected Package getPackage(String name);

    @Positive
    @CFComment({ "nullness: The size of array passed to toArray", "method is of exact same size as of the map for which toArray method is invoked" })
    @Positive
    @SuppressWarnings({ "nullness:return" })
    @Positive
    protected Package[] getPackages();

    @Positive
    Stream<Package> packages();

    @Positive
    @Nullable
    @Positive
    protected String findLibrary(String libname);

    @Positive
    @CFComment({ "nulness: usr_paths and sys_paths are initialized", "by intializePath method if they are null" })
    @Positive
    @SuppressWarnings({ "nullness:dereference.of.nullable" })
    @Positive
    static NativeLibrary loadLibrary(Class<?> fromClass, File file);

    @Positive
    static NativeLibrary loadLibrary(Class<?> fromClass, String name);

    @Positive
    static long findNative(@Nullable ClassLoader loader, String entryName);

    @Positive
    public void setDefaultAssertionStatus(boolean enabled);

    @Positive
    public void setPackageAssertionStatus(@Nullable String packageName, boolean enabled);

    @Positive
    public void setClassAssertionStatus(String className, boolean enabled);

    @Positive
    public void clearAssertionStatus();

    @Positive
    @RequiresNonNull({ "classAssertionStatus", "packageAssertionStatus" })
    @Positive
    boolean desiredAssertionStatus(String className);

    @Positive
    ConcurrentHashMap<?, ?> createOrGetClassLoaderValueMap();
    @Positive
}

    @Positive
final class CompoundEnumeration<E> implements Enumeration<E> {

    @Positive
    public CompoundEnumeration(Enumeration<E>[] enums) {
    @Positive
    }

    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean hasMoreElements();

    @Positive
    public E nextElement(@NonEmpty CompoundEnumeration<E> this);
    @Positive
}

// CFWR semantic augmentation - variant 1
