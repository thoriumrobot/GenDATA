/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.beans;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.io.Serial;
import java.io.Serializable;
import java.util.Hashtable;
import java.util.Map.Entry;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class VetoableChangeSupport implements Serializable {

    public VetoableChangeSupport(Object sourceBean) {
    }

    public void addVetoableChangeListener(VetoableChangeListener listener);

    public void removeVetoableChangeListener(VetoableChangeListener listener);

    public VetoableChangeListener[] getVetoableChangeListeners();

    public void addVetoableChangeListener(String propertyName, VetoableChangeListener listener);

    public void removeVetoableChangeListener(String propertyName, VetoableChangeListener listener);

    public VetoableChangeListener[] getVetoableChangeListeners(String propertyName);

    public void fireVetoableChange(String propertyName, Object oldValue, Object newValue) throws PropertyVetoException;

    public void fireVetoableChange(String propertyName, int oldValue, int newValue) throws PropertyVetoException;

    public void fireVetoableChange(String propertyName, boolean oldValue, boolean newValue) throws PropertyVetoException;

    public void fireVetoableChange(PropertyChangeEvent event) throws PropertyVetoException;

    public boolean hasListeners(String propertyName);

    private static final class VetoableChangeListenerMap extends ChangeListenerMap<VetoableChangeListener> {

        @Override
        protected VetoableChangeListener[] newArray(int length);

        @Override
        protected VetoableChangeListener newProxy(String name, VetoableChangeListener listener);

        public VetoableChangeListener extract(VetoableChangeListener listener);
    }
}
