/*
    @Positive
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.beans.beancontext;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.beans.PropertyChangeSupport;
    @Positive
import java.beans.PropertyVetoException;
    @Positive
import java.beans.VetoableChangeListener;
    @Positive
import java.beans.VetoableChangeSupport;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class BeanContextChildSupport implements BeanContextChild, BeanContextServicesListener, Serializable {

    @Positive
    public BeanContextChildSupport() {
    @Positive
    }

    @Positive
    public BeanContextChildSupport(BeanContextChild bcc) {
    @Positive
    }

    @Positive
    public synchronized void setBeanContext(BeanContext bc) throws PropertyVetoException;

    @Positive
    public synchronized BeanContext getBeanContext();

    @Positive
    public void addPropertyChangeListener(String name, PropertyChangeListener pcl);

    @Positive
    public void removePropertyChangeListener(String name, PropertyChangeListener pcl);

    @Positive
    public void addVetoableChangeListener(String name, VetoableChangeListener vcl);

    @Positive
    public void removeVetoableChangeListener(String name, VetoableChangeListener vcl);

    @Positive
    public void serviceRevoked(BeanContextServiceRevokedEvent bcsre);

    @Positive
    public void serviceAvailable(BeanContextServiceAvailableEvent bcsae);

    @Positive
    public BeanContextChild getBeanContextChildPeer();

    @Positive
    public boolean isDelegated();

    @Positive
    public void firePropertyChange(String name, Object oldValue, Object newValue);

    @Positive
    public void fireVetoableChange(String name, Object oldValue, Object newValue) throws PropertyVetoException;

    @Positive
    public boolean validatePendingSetBeanContext(BeanContext newValue);

    @Positive
    protected void releaseBeanContextResources();

    @Positive
    protected void initializeBeanContextResources();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public BeanContextChild beanContextChildPeer;

    @Positive
    protected PropertyChangeSupport pcSupport;

    @Positive
    protected VetoableChangeSupport vcSupport;

    @Positive
    protected transient BeanContext beanContext;

    @Positive
    protected transient boolean rejectedSetBCOnce;
    @Positive
}

// CFWR semantic augmentation - variant 1
