/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.Closeable;
    @Positive
import java.io.File;
    @Positive
import java.io.FilePermission;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.CodeSigner;
    @Positive
import java.security.CodeSource;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PermissionCollection;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.security.SecureClassLoader;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.jar.Attributes;
    @Positive
import java.util.jar.Attributes.Name;
    @Positive
import java.util.jar.JarFile;
    @Positive
import java.util.jar.Manifest;
    @Positive
import jdk.internal.loader.Resource;
    @Positive
import jdk.internal.loader.URLClassPath;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.perf.PerfCounter;
    @Positive
import sun.net.www.ParseUtil;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public class URLClassLoader extends SecureClassLoader implements Closeable {

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public URLClassLoader(URL[] urls, @Nullable ClassLoader parent) {
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public URLClassLoader(URL[] urls) {
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public URLClassLoader(URL[] urls, ClassLoader parent, URLStreamHandlerFactory factory) {
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public URLClassLoader(String name, URL[] urls, ClassLoader parent) {
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public URLClassLoader(String name, URL[] urls, ClassLoader parent, URLStreamHandlerFactory factory) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public InputStream getResourceAsStream(String name);

    @Positive
    public void close() throws IOException;

    @Positive
    protected void addURL(@Nullable URL url);

    @Positive
    public URL[] getURLs();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    protected Class<?> findClass(final String name) throws ClassNotFoundException;

    @Positive
    protected Package definePackage(String name, Manifest man, @Nullable URL url);

    @Positive
    @Nullable
    @Positive
    public URL findResource(final String name);

    @Positive
    public Enumeration<URL> findResources(final String name) throws IOException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    protected PermissionCollection getPermissions(CodeSource codesource);

    @Positive
    public static URLClassLoader newInstance(final URL[] urls, final ClassLoader parent);

    @Positive
    public static URLClassLoader newInstance(final URL[] urls);
    @Positive
}

    @Positive
final class FactoryURLClassLoader extends URLClassLoader {

    @Positive
    public final Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException;
    @Positive
}

// CFWR semantic augmentation - variant 1
