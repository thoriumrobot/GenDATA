/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.reflect;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.invoke.MethodHandle;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.MethodType;
    @Positive
import java.lang.invoke.WrongMethodTypeException;
    @Positive
import java.lang.module.ModuleDescriptor;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayDeque;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Deque;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.IdentityHashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import java.util.function.BooleanSupplier;
    @Positive
import jdk.internal.access.JavaLangAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.module.Modules;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.loader.ClassLoaderValue;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.reflect.misc.ReflectUtil;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import static java.lang.invoke.MethodType.methodType;
    @Positive
import static java.lang.module.ModuleDescriptor.Modifier.SYNTHETIC;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class Proxy implements java.io.Serializable {

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected InvocationHandler h;

    @Positive
    protected Proxy(InvocationHandler h) {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    @CallerSensitive
    @Positive
    public static Class<?> getProxyClass(@Nullable ClassLoader loader, Class<?>... interfaces) throws IllegalArgumentException;

    @Positive
    private static final class ProxyBuilder {

    @Positive
        static boolean isProxyClass(Class<?> c);

    @Positive
        static void trace(String cn, Module module, ClassLoader loader, List<Class<?>> interfaces);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        Constructor<?> build();
    @Positive
    }

    @Positive
    @CallerSensitive
    @Positive
    public static Object newProxyInstance(@Nullable ClassLoader loader, Class<?>[] interfaces, InvocationHandler h);

    @Positive
    public static boolean isProxyClass(Class<?> cl);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @CallerSensitive
    @Positive
    public static InvocationHandler getInvocationHandler(Object proxy) throws IllegalArgumentException;

    @Positive
    static MethodHandle defaultMethodHandle(Class<? extends Proxy> proxyClass, Method method);

    @Positive
    static class InvocationException extends ReflectiveOperationException {

    @Positive
        static Object wrap(Throwable cause) throws InvocationException;

    @Positive
        static MethodHandle wrapMH();
    @Positive
    }
    @Positive
}
