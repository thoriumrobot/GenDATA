/*
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.accessibility;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.IllegalComponentStateException;
import java.beans.BeanProperty;
import java.beans.JavaBean;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.util.Locale;
import sun.awt.AWTAccessor;
import sun.awt.AppContext;

@AnnotatedFor({ "interning" })
@JavaBean(description = "Minimal information that all accessible objects return")
public abstract class AccessibleContext {

    protected AccessibleContext() {
    }

    @Interned
    public static final String ACCESSIBLE_NAME_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_DESCRIPTION_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_STATE_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_VALUE_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_SELECTION_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_CARET_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_VISIBLE_DATA_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_CHILD_PROPERTY;

    @Interned
    public static final String ACCESSIBLE_ACTIVE_DESCENDANT_PROPERTY;

    public static final String ACCESSIBLE_TABLE_CAPTION_CHANGED;

    public static final String ACCESSIBLE_TABLE_SUMMARY_CHANGED;

    public static final String ACCESSIBLE_TABLE_MODEL_CHANGED;

    public static final String ACCESSIBLE_TABLE_ROW_HEADER_CHANGED;

    public static final String ACCESSIBLE_TABLE_ROW_DESCRIPTION_CHANGED;

    public static final String ACCESSIBLE_TABLE_COLUMN_HEADER_CHANGED;

    public static final String ACCESSIBLE_TABLE_COLUMN_DESCRIPTION_CHANGED;

    public static final String ACCESSIBLE_ACTION_PROPERTY;

    public static final String ACCESSIBLE_HYPERTEXT_OFFSET;

    public static final String ACCESSIBLE_TEXT_PROPERTY;

    public static final String ACCESSIBLE_INVALIDATE_CHILDREN;

    public static final String ACCESSIBLE_TEXT_ATTRIBUTES_CHANGED;

    public static final String ACCESSIBLE_COMPONENT_BOUNDS_CHANGED;

    protected Accessible accessibleParent;

    protected String accessibleName;

    protected String accessibleDescription;

    public String getAccessibleName();

    @BeanProperty(preferred = true, description = "Sets the accessible name for the component.")
    public void setAccessibleName(String s);

    public String getAccessibleDescription();

    @BeanProperty(preferred = true, description = "Sets the accessible description for the component.")
    public void setAccessibleDescription(String s);

    public abstract AccessibleRole getAccessibleRole();

    public abstract AccessibleStateSet getAccessibleStateSet();

    public Accessible getAccessibleParent();

    public void setAccessibleParent(Accessible a);

    public abstract int getAccessibleIndexInParent();

    public abstract int getAccessibleChildrenCount();

    public abstract Accessible getAccessibleChild(int i);

    public abstract Locale getLocale() throws IllegalComponentStateException;

    public void addPropertyChangeListener(PropertyChangeListener listener);

    public void removePropertyChangeListener(PropertyChangeListener listener);

    public AccessibleAction getAccessibleAction();

    public AccessibleComponent getAccessibleComponent();

    public AccessibleSelection getAccessibleSelection();

    public AccessibleText getAccessibleText();

    public AccessibleEditableText getAccessibleEditableText();

    public AccessibleValue getAccessibleValue();

    public AccessibleIcon[] getAccessibleIcon();

    public AccessibleRelationSet getAccessibleRelationSet();

    public AccessibleTable getAccessibleTable();

    public void firePropertyChange(String propertyName, Object oldValue, Object newValue);
}
