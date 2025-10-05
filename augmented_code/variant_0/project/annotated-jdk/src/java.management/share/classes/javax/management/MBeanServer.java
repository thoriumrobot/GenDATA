/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.util.Set;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import javax.management.loading.ClassLoaderRepository;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public interface MBeanServer extends MBeanServerConnection {

    @Positive
    public ObjectInstance createMBean(String className, @Nullable ObjectName name) throws ReflectionException, InstanceAlreadyExistsException, MBeanRegistrationException, MBeanException, NotCompliantMBeanException;

    @Positive
    public ObjectInstance createMBean(String className, @Nullable ObjectName name, ObjectName loaderName) throws ReflectionException, InstanceAlreadyExistsException, MBeanRegistrationException, MBeanException, NotCompliantMBeanException, InstanceNotFoundException;

    @Positive
    public ObjectInstance createMBean(String className, @Nullable ObjectName name, Object @Nullable [] params, String @Nullable [] signature) throws ReflectionException, InstanceAlreadyExistsException, MBeanRegistrationException, MBeanException, NotCompliantMBeanException;

    @Positive
    public ObjectInstance createMBean(String className, @Nullable ObjectName name, ObjectName loaderName, Object @Nullable [] params, String @Nullable [] signature) throws ReflectionException, InstanceAlreadyExistsException, MBeanRegistrationException, MBeanException, NotCompliantMBeanException, InstanceNotFoundException;

    @Positive
    public ObjectInstance registerMBean(Object object, @Nullable ObjectName name) throws InstanceAlreadyExistsException, MBeanRegistrationException, NotCompliantMBeanException;

    @Positive
    public void unregisterMBean(@Nullable ObjectName name) throws InstanceNotFoundException, MBeanRegistrationException;

    @Positive
    public ObjectInstance getObjectInstance(@Nullable ObjectName name) throws InstanceNotFoundException;

    @Positive
    public Set<ObjectInstance> queryMBeans(@Nullable ObjectName name, @Nullable QueryExp query);

    @Positive
    public Set<ObjectName> queryNames(@Nullable ObjectName name, @Nullable QueryExp query);

    @Positive
    public boolean isRegistered(ObjectName name);

    @Positive
    public Integer getMBeanCount();

    @Positive
    public Object getAttribute(ObjectName name, String attribute) throws MBeanException, AttributeNotFoundException, InstanceNotFoundException, ReflectionException;

    @Positive
    public AttributeList getAttributes(ObjectName name, String[] attributes) throws InstanceNotFoundException, ReflectionException;

    @Positive
    public void setAttribute(ObjectName name, Attribute attribute) throws InstanceNotFoundException, AttributeNotFoundException, InvalidAttributeValueException, MBeanException, ReflectionException;

    @Positive
    public AttributeList setAttributes(ObjectName name, AttributeList attributes) throws InstanceNotFoundException, ReflectionException;

    @Positive
    public Object invoke(@Nullable ObjectName name, String operationName, Object @Nullable [] params, String @Nullable [] signature) throws InstanceNotFoundException, MBeanException, ReflectionException;

    @Positive
    public String getDefaultDomain();

    @Positive
    public String[] getDomains();

    @Positive
    public void addNotificationListener(@Nullable ObjectName name, NotificationListener listener, @Nullable NotificationFilter filter, Object handback) throws InstanceNotFoundException;

    @Positive
    public void addNotificationListener(@Nullable ObjectName name, ObjectName listener, @Nullable NotificationFilter filter, Object handback) throws InstanceNotFoundException;

    @Positive
    public void removeNotificationListener(@Nullable ObjectName name, ObjectName listener) throws InstanceNotFoundException, ListenerNotFoundException;

    @Positive
    public void removeNotificationListener(@Nullable ObjectName name, ObjectName listener, @Nullable NotificationFilter filter, @Nullable Object handback) throws InstanceNotFoundException, ListenerNotFoundException;

    @Positive
    public void removeNotificationListener(@Nullable ObjectName name, @Nullable NotificationListener listener) throws InstanceNotFoundException, ListenerNotFoundException;

    @Positive
    public void removeNotificationListener(@Nullable ObjectName name, NotificationListener listener, @Nullable NotificationFilter filter, @Nullable Object handback) throws InstanceNotFoundException, ListenerNotFoundException;

    @Positive
    public MBeanInfo getMBeanInfo(@Nullable ObjectName name) throws InstanceNotFoundException, IntrospectionException, ReflectionException;

    @Positive
    public boolean isInstanceOf(@Nullable ObjectName name, String className) throws InstanceNotFoundException;

    @Positive
    public Object instantiate(String className) throws ReflectionException, MBeanException;

    @Positive
    public Object instantiate(String className, @Nullable ObjectName loaderName) throws ReflectionException, MBeanException, InstanceNotFoundException;

    @Positive
    public Object instantiate(String className, Object @Nullable [] params, String @Nullable [] signature) throws ReflectionException, MBeanException;

    @Positive
    public Object instantiate(String className, @Nullable ObjectName loaderName, Object @Nullable [] params, String @Nullable [] signature) throws ReflectionException, MBeanException, InstanceNotFoundException;

    @Positive
    @Deprecated()
    @Positive
    default public ObjectInputStream deserialize(@Nullable ObjectName name, byte[] data) throws InstanceNotFoundException, OperationsException;

    @Positive
    @Deprecated()
    @Positive
    default public ObjectInputStream deserialize(String className, byte[] data) throws OperationsException, ReflectionException;

    @Positive
    @Deprecated()
    @Positive
    default public ObjectInputStream deserialize(String className, @Nullable ObjectName loaderName, byte[] data) throws InstanceNotFoundException, OperationsException, ReflectionException;

    @Positive
    public ClassLoader getClassLoaderFor(ObjectName mbeanName) throws InstanceNotFoundException;

    @Positive
    public ClassLoader getClassLoader(@Nullable ObjectName loaderName) throws InstanceNotFoundException;

    @Positive
    public ClassLoaderRepository getClassLoaderRepository();
    @Positive
}
