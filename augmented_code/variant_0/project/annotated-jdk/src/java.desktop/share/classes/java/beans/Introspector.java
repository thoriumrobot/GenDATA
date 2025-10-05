/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package java.beans;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Component;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Type;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.EventObject;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.TreeMap;
    @Positive
import com.sun.beans.TypeResolver;
    @Positive
import com.sun.beans.finder.ClassFinder;
    @Positive
import com.sun.beans.introspect.ClassInfo;
    @Positive
import com.sun.beans.introspect.EventSetInfo;
    @Positive
import com.sun.beans.introspect.PropertyInfo;
    @Positive
import jdk.internal.access.JavaBeansAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Introspector {

    @Positive
    public static final int USE_ALL_BEANINFO;

    @Positive
    public static final int IGNORE_IMMEDIATE_BEANINFO;

    @Positive
    public static final int IGNORE_ALL_BEANINFO;

    @Positive
    public static BeanInfo getBeanInfo(Class<?> beanClass) throws IntrospectionException;

    @Positive
    public static BeanInfo getBeanInfo(Class<?> beanClass, int flags) throws IntrospectionException;

    @Positive
    public static BeanInfo getBeanInfo(Class<?> beanClass, Class<?> stopClass) throws IntrospectionException;

    @Positive
    public static BeanInfo getBeanInfo(Class<?> beanClass, Class<?> stopClass, int flags) throws IntrospectionException;

    @Positive
    public static String decapitalize(String name);

    @Positive
    public static String[] getBeanInfoSearchPath();

    @Positive
    public static void setBeanInfoSearchPath(String[] path);

    @Positive
    public static void flushCaches();

    @Positive
    public static void flushFromCaches(Class<?> clz);

    @Positive
    static Method findMethod(Class<?> cls, String methodName, int argCount);

    @Positive
    static Method findMethod(Class<?> cls, String methodName, int argCount, Class<?>[] args);

    @Positive
    static boolean isSubclass(Class<?> a, Class<?> b);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    static Object instantiate(Class<?> sibling, String className) throws InstantiationException, IllegalAccessException, NoSuchMethodException, InvocationTargetException, ClassNotFoundException;
    @Positive
}

    @Positive
class GenericBeanInfo extends SimpleBeanInfo {

    @Positive
    public GenericBeanInfo(BeanDescriptor beanDescriptor, EventSetDescriptor[] events, int defaultEvent, PropertyDescriptor[] properties, int defaultProperty, MethodDescriptor[] methods, BeanInfo targetBeanInfo) {
    @Positive
    }

    @Positive
    public PropertyDescriptor[] getPropertyDescriptors();

    @Positive
    public int getDefaultPropertyIndex();

    @Positive
    public EventSetDescriptor[] getEventSetDescriptors();

    @Positive
    public int getDefaultEventIndex();

    @Positive
    public MethodDescriptor[] getMethodDescriptors();

    @Positive
    public BeanDescriptor getBeanDescriptor();

    @Positive
    public java.awt.Image getIcon(int iconKind);
    @Positive
}
