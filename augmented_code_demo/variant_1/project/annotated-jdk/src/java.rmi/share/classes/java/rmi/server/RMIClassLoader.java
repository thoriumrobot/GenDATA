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
package java.rmi.server;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URL;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.ServiceLoader;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class RMIClassLoader {

    @Positive
    @Deprecated
    @Positive
    public static Class<?> loadClass(String name) throws MalformedURLException, ClassNotFoundException;

    @Positive
    public static Class<?> loadClass(URL codebase, String name) throws MalformedURLException, ClassNotFoundException;

    @Positive
    public static Class<?> loadClass(String codebase, String name) throws MalformedURLException, ClassNotFoundException;

    @Positive
    public static Class<?> loadClass(String codebase, String name, ClassLoader defaultLoader) throws MalformedURLException, ClassNotFoundException;

    @Positive
    public static Class<?> loadProxyClass(String codebase, String[] interfaces, ClassLoader defaultLoader) throws ClassNotFoundException, MalformedURLException;

    @Positive
    public static ClassLoader getClassLoader(String codebase) throws MalformedURLException, SecurityException;

    @Positive
    public static String getClassAnnotation(Class<?> cl);

    @Positive
    public static RMIClassLoaderSpi getDefaultProviderInstance();

    @Positive
    @Deprecated
    @Positive
    public static Object getSecurityContext(ClassLoader loader);
    @Positive
}

// CFWR semantic augmentation - variant 1
