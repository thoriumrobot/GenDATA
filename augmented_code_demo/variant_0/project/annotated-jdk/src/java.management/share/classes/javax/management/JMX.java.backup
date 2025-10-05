/*
 * Copyright (c) 2005, 2013, Oracle and/or its affiliates. All rights reserved.
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
package javax.management;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import com.sun.jmx.mbeanserver.Introspector;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Modifier;
import java.lang.reflect.Proxy;
import sun.reflect.misc.ReflectUtil;

@AnnotatedFor({ "interning" })
public class JMX {

    @Interned
    public static final String DEFAULT_VALUE_FIELD;

    @Interned
    public static final String IMMUTABLE_INFO_FIELD;

    @Interned
    public static final String INTERFACE_CLASS_NAME_FIELD;

    @Interned
    public static final String LEGAL_VALUES_FIELD;

    @Interned
    public static final String MAX_VALUE_FIELD;

    @Interned
    public static final String MIN_VALUE_FIELD;

    @Interned
    public static final String MXBEAN_FIELD;

    @Interned
    public static final String OPEN_TYPE_FIELD;

    @Interned
    public static final String ORIGINAL_TYPE_FIELD;

    public static <T> T newMBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass);

    public static <T> T newMBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass, boolean notificationEmitter);

    public static <T> T newMXBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass);

    public static <T> T newMXBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass, boolean notificationEmitter);

    public static boolean isMXBeanInterface(Class<?> interfaceClass);
}
