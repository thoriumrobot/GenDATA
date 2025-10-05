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
package java.rmi.server;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.net.MalformedURLException;
import java.net.URL;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Iterator;
import java.util.ServiceLoader;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class RMIClassLoader {

    @Deprecated
    public static Class<?> loadClass(String name) throws MalformedURLException, ClassNotFoundException;

    public static Class<?> loadClass(URL codebase, String name) throws MalformedURLException, ClassNotFoundException;

    public static Class<?> loadClass(String codebase, String name) throws MalformedURLException, ClassNotFoundException;

    public static Class<?> loadClass(String codebase, String name, ClassLoader defaultLoader) throws MalformedURLException, ClassNotFoundException;

    public static Class<?> loadProxyClass(String codebase, String[] interfaces, ClassLoader defaultLoader) throws ClassNotFoundException, MalformedURLException;

    public static ClassLoader getClassLoader(String codebase) throws MalformedURLException, SecurityException;

    public static String getClassAnnotation(Class<?> cl);

    public static RMIClassLoaderSpi getDefaultProviderInstance();

    @Deprecated
    public static Object getSecurityContext(ClassLoader loader);
}
