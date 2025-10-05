/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.reflect;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.WrongMethodTypeException;
import java.lang.module.ModuleDescriptor;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import jdk.internal.access.JavaLangAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.module.Modules;
import jdk.internal.misc.VM;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import jdk.internal.loader.ClassLoaderValue;
import jdk.internal.vm.annotation.Stable;
import sun.reflect.misc.ReflectUtil;
import sun.security.action.GetPropertyAction;
import sun.security.util.SecurityConstants;
import static java.lang.invoke.MethodType.methodType;
import static java.lang.module.ModuleDescriptor.Modifier.SYNTHETIC;

@AnnotatedFor({ "nullness" })
public class Proxy implements java.io.Serializable {

    @SuppressWarnings("serial")
    protected InvocationHandler h;

    protected Proxy(InvocationHandler h) {
    }

    @Deprecated
    @CallerSensitive
    public static Class<?> getProxyClass(@Nullable ClassLoader loader, Class<?>... interfaces) throws IllegalArgumentException;

    private static final class ProxyBuilder {

        static boolean isProxyClass(Class<?> c);

        static void trace(String cn, Module module, ClassLoader loader, List<Class<?>> interfaces);

        @SuppressWarnings("removal")
        Constructor<?> build();
    }

    @CallerSensitive
    public static Object newProxyInstance(@Nullable ClassLoader loader, Class<?>[] interfaces, InvocationHandler h);

    public static boolean isProxyClass(Class<?> cl);

    @SuppressWarnings("removal")
    @CallerSensitive
    public static InvocationHandler getInvocationHandler(Object proxy) throws IllegalArgumentException;

    static MethodHandle defaultMethodHandle(Class<? extends Proxy> proxyClass, Method method);

    static class InvocationException extends ReflectiveOperationException {

        static Object wrap(Throwable cause) throws InvocationException;

        static MethodHandle wrapMH();
    }
}
