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
import org.checkerframework.checker.fenum.qual.FenumTop;
    @Positive
import org.checkerframework.checker.guieffect.qual.PolyUI;
    @Positive
import org.checkerframework.checker.guieffect.qual.PolyUIEffect;
    @Positive
import org.checkerframework.checker.guieffect.qual.PolyUIType;
    @Positive
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.initialization.qual.NotOnlyInitialized;
    @Positive
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Map.Entry;

    @Positive
@AnnotatedFor({ "fenum", "guieffect", "interning", "nullness" })
    @Positive
@PolyUIType
    @Positive
@UsesObjectEquals
    @Positive
public class PropertyChangeSupport implements Serializable {

    @Positive
    @SafeEffect
    @Positive
    public PropertyChangeSupport(@PolyUI @UnknownInitialization(Object.class) Object sourceBean) {
    @Positive
    }

    @Positive
    @PolyUIEffect
    @Positive
    public void addPropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable @PolyUI PropertyChangeListener listener);

    @Positive
    @PolyUIEffect
    @Positive
    public void removePropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable PropertyChangeListener listener);

    @Positive
    @PolyUIEffect
    @Positive
    @PolyUI
    @Positive
    public PropertyChangeListener[] getPropertyChangeListeners(@PolyUI PropertyChangeSupport this);

    @Positive
    @PolyUIEffect
    @Positive
    public void addPropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable String propertyName, @Nullable @PolyUI PropertyChangeListener listener);

    @Positive
    @PolyUIEffect
    @Positive
    public void removePropertyChangeListener(@PolyUI PropertyChangeSupport this, @Nullable String propertyName, @Nullable PropertyChangeListener listener);

    @Positive
    @PolyUIEffect
    @Positive
    @PolyUI
    @Positive
    public PropertyChangeListener[] getPropertyChangeListeners(@PolyUI PropertyChangeSupport this, String propertyName);

    @Positive
    @PolyUIEffect
    @Positive
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, @Nullable @FenumTop Object oldValue, @Nullable @FenumTop Object newValue);

    @Positive
    @PolyUIEffect
    @Positive
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, @FenumTop int oldValue, @FenumTop int newValue);

    @Positive
    @PolyUIEffect
    @Positive
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, boolean oldValue, boolean newValue);

    @Positive
    @PolyUIEffect
    @Positive
    public void firePropertyChange(@PolyUI PropertyChangeSupport this, PropertyChangeEvent event);

    @Positive
    @PolyUIEffect
    @Positive
    public void fireIndexedPropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, int index, @Nullable Object oldValue, @Nullable Object newValue);

    @Positive
    @PolyUIEffect
    @Positive
    public void fireIndexedPropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, int index, int oldValue, int newValue);

    @Positive
    @PolyUIEffect
    @Positive
    public void fireIndexedPropertyChange(@PolyUI PropertyChangeSupport this, String propertyName, int index, boolean oldValue, boolean newValue);

    @Positive
    @PolyUIEffect
    @Positive
    public boolean hasListeners(@PolyUI PropertyChangeSupport this, @Nullable String propertyName);

    @Positive
    private static final class PropertyChangeListenerMap extends ChangeListenerMap<PropertyChangeListener> {

    @Positive
        @Override
    @Positive
        protected PropertyChangeListener[] newArray(int length);

    @Positive
        @Override
    @Positive
        protected PropertyChangeListener newProxy(String name, PropertyChangeListener listener);

    @Positive
        public PropertyChangeListener extract(PropertyChangeListener listener);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
