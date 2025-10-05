/*
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.beans.beancontext;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.beans.PropertyVetoException;
import java.beans.VetoableChangeListener;
import java.beans.VetoableChangeSupport;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class BeanContextChildSupport implements BeanContextChild, BeanContextServicesListener, Serializable {

    public BeanContextChildSupport() {
    }

    public BeanContextChildSupport(BeanContextChild bcc) {
    }

    public synchronized void setBeanContext(BeanContext bc) throws PropertyVetoException;

    public synchronized BeanContext getBeanContext();

    public void addPropertyChangeListener(String name, PropertyChangeListener pcl);

    public void removePropertyChangeListener(String name, PropertyChangeListener pcl);

    public void addVetoableChangeListener(String name, VetoableChangeListener vcl);

    public void removeVetoableChangeListener(String name, VetoableChangeListener vcl);

    public void serviceRevoked(BeanContextServiceRevokedEvent bcsre);

    public void serviceAvailable(BeanContextServiceAvailableEvent bcsae);

    public BeanContextChild getBeanContextChildPeer();

    public boolean isDelegated();

    public void firePropertyChange(String name, Object oldValue, Object newValue);

    public void fireVetoableChange(String name, Object oldValue, Object newValue) throws PropertyVetoException;

    public boolean validatePendingSetBeanContext(BeanContext newValue);

    protected void releaseBeanContextResources();

    protected void initializeBeanContextResources();

    @SuppressWarnings("serial")
    public BeanContextChild beanContextChildPeer;

    protected PropertyChangeSupport pcSupport;

    protected VetoableChangeSupport vcSupport;

    protected transient BeanContext beanContext;

    protected transient boolean rejectedSetBCOnce;
}
