/*
    @Positive
 * Copyright (c) 2005, 2013, Oracle and/or its affiliates. All rights reserved.
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
package javax.management;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import com.sun.jmx.mbeanserver.Introspector;
    @Positive
import java.lang.reflect.InvocationHandler;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.lang.reflect.Proxy;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class JMX {

    @Positive
    @Interned
    @Positive
    public static final String DEFAULT_VALUE_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String IMMUTABLE_INFO_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String INTERFACE_CLASS_NAME_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String LEGAL_VALUES_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String MAX_VALUE_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String MIN_VALUE_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String MXBEAN_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String OPEN_TYPE_FIELD;

    @Positive
    @Interned
    @Positive
    public static final String ORIGINAL_TYPE_FIELD;

    @Positive
    public static <T> T newMBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass);

    @Positive
    public static <T> T newMBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass, boolean notificationEmitter);

    @Positive
    public static <T> T newMXBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass);

    @Positive
    public static <T> T newMXBeanProxy(MBeanServerConnection connection, ObjectName objectName, Class<T> interfaceClass, boolean notificationEmitter);

    @Positive
    public static boolean isMXBeanInterface(Class<?> interfaceClass);
    @Positive
}

// CFWR semantic augmentation - variant 1
