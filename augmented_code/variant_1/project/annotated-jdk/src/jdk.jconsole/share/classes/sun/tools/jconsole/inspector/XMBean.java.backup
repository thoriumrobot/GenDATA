/*
 * Copyright (c) 2004, 2008, Oracle and/or its affiliates. All rights reserved.
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
package sun.tools.jconsole.inspector;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import javax.management.*;
import javax.swing.Icon;
import sun.tools.jconsole.JConsole;
import sun.tools.jconsole.MBeansTab;
import sun.tools.jconsole.ProxyClient.SnapshotMBeanServerConnection;

public class XMBean {

    public XMBean(ObjectName objectName, MBeansTab mbeansTab) {
    }

    MBeanServerConnection getMBeanServerConnection();

    SnapshotMBeanServerConnection getSnapshotMBeanServerConnection();

    public Boolean isBroadcaster();

    public Object invoke(String operationName) throws Exception;

    public Object invoke(String operationName, Object[] params, String[] sig) throws Exception;

    public void setAttribute(Attribute attribute) throws AttributeNotFoundException, InstanceNotFoundException, InvalidAttributeValueException, MBeanException, ReflectionException, IOException;

    public Object getAttribute(String attributeName) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    public AttributeList getAttributes(String[] attributeNames) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    public AttributeList getAttributes(MBeanAttributeInfo[] attributeNames) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    public ObjectName getObjectName();

    public MBeanInfo getMBeanInfo() throws InstanceNotFoundException, IntrospectionException, ReflectionException, IOException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    public String getText();

    public void setText(String text);

    public Icon getIcon();

    public void setIcon(Icon icon);

    @Override
    public String toString();
}
