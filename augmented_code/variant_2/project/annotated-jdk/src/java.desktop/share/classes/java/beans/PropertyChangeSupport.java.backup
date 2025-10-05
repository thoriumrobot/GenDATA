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

import org.checkerframework.checker.fenum.qual.FenumTop;
import org.checkerframework.checker.guieffect.qual.PolyUI;
import org.checkerframework.checker.guieffect.qual.PolyUIEffect;
import org.checkerframework.checker.guieffect.qual.PolyUIType;
import org.checkerframework.checker.guieffect.qual.SafeEffect;
import org.checkerframework.checker.initialization.qual.NotOnlyInitialized;
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.io.Serial;
import java.io.Serializable;
import java.util.Hashtable;
import java.util.Map.Entry;

@AnnotatedFor({ "fenum", "guieffect", "interning", "nullness" })
@PolyUIType
@UsesObjectEquals
public class PropertyChangeSupport implements Serializable {

    @SafeEffect
    public PropertyChangeSupport(@PolyUI @UnknownInitialization(Object.class) Object sourceBean) {
    }

    @PolyUIEffect
    public void addPropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable @PolyUI PropertyChangeListener listener);

    @PolyUIEffect
    public void removePropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable PropertyChangeListener listener);

    @PolyUIEffect
    @PolyUI
    public PropertyChangeListener[] getPropertyChangeListeners(@PolyUI PropertyChangeSupport this);

    @PolyUIEffect
    public void addPropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable String propertyName, @Nullable @PolyUI PropertyChangeListener listener);

    @PolyUIEffect
    public void removePropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable String propertyName, @Nullable PropertyChangeListener listener);

    @PolyUIEffect
    @PolyUI
    public PropertyChangeListener[] getPropertyChangeListeners(@PolyUI PropertyChangeSupport this, String propertyName);

    @PolyUIEffect
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, @Nullable @FenumTop Object oldValue, @Nullable @FenumTop Object newValue);

    @PolyUIEffect
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, @FenumTop int oldValue, @FenumTop int newValue);

    @PolyUIEffect
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, boolean oldValue, boolean newValue);

    @PolyUIEffect
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, PropertyChangeEvent event);

    @PolyUIEffect
    public void fireIndexedPropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, int index, @Nullable Object oldValue, @Nullable Object newValue);

    @PolyUIEffect
    public void fireIndexedPropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, int index, int oldValue, int newValue);

    @PolyUIEffect
    public void fireIndexedPropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, int index, boolean oldValue, boolean newValue);

    @PolyUIEffect
    public boolean hasListeners(@PolyUI PropertyChangeSupport this, @Nullable String propertyName);

    private static final class PropertyChangeListenerMap extends ChangeListenerMap<PropertyChangeListener> {

        @Override
        protected PropertyChangeListener[] newArray(int length);

        @Override
        protected PropertyChangeListener newProxy(String name, PropertyChangeListener listener);

        public PropertyChangeListener extract(PropertyChangeListener listener);
    }
}
