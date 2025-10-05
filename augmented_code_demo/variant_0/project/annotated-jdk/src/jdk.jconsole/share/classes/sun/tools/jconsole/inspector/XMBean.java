/*
    @Positive
 * Copyright (c) 2004, 2008, Oracle and/or its affiliates. All rights reserved.
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
package sun.tools.jconsole.inspector;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.IOException;
    @Positive
import javax.management.*;
    @Positive
import javax.swing.Icon;
    @Positive
import sun.tools.jconsole.JConsole;
    @Positive
import sun.tools.jconsole.MBeansTab;
    @Positive
import sun.tools.jconsole.ProxyClient.SnapshotMBeanServerConnection;

    @Positive
public class XMBean {

    @Positive
    public XMBean(ObjectName objectName, MBeansTab mbeansTab) {
    @Positive
    }

    @Positive
    MBeanServerConnection getMBeanServerConnection();

    @Positive
    SnapshotMBeanServerConnection getSnapshotMBeanServerConnection();

    @Positive
    public Boolean isBroadcaster();

    @Positive
    public Object invoke(String operationName) throws Exception;

    @Positive
    public Object invoke(String operationName, Object[] params, String[] sig) throws Exception;

    @Positive
    public void setAttribute(Attribute attribute) throws AttributeNotFoundException, InstanceNotFoundException, InvalidAttributeValueException, MBeanException, ReflectionException, IOException;

    @Positive
    public Object getAttribute(String attributeName) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    @Positive
    public AttributeList getAttributes(String[] attributeNames) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    @Positive
    public AttributeList getAttributes(MBeanAttributeInfo[] attributeNames) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    @Positive
    public ObjectName getObjectName();

    @Positive
    public MBeanInfo getMBeanInfo() throws InstanceNotFoundException, IntrospectionException, ReflectionException, IOException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    public String getText();

    @Positive
    public void setText(String text);

    @Positive
    public Icon getIcon();

    @Positive
    public void setIcon(Icon icon);

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
