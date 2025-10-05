/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.accessibility;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.IllegalComponentStateException;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.beans.PropertyChangeSupport;
    @Positive
import java.util.Locale;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AppContext;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@JavaBean(description = "Minimal information that all accessible objects return")
    @Positive
public abstract class AccessibleContext {

    @Positive
    protected AccessibleContext() {
    @Positive
    }

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_NAME_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_DESCRIPTION_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_STATE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_VALUE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_SELECTION_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_CARET_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_VISIBLE_DATA_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_CHILD_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSIBLE_ACTIVE_DESCENDANT_PROPERTY;

    @Positive
    public static final String ACCESSIBLE_TABLE_CAPTION_CHANGED;

    @Positive
    public static final String ACCESSIBLE_TABLE_SUMMARY_CHANGED;

    @Positive
    public static final String ACCESSIBLE_TABLE_MODEL_CHANGED;

    @Positive
    public static final String ACCESSIBLE_TABLE_ROW_HEADER_CHANGED;

    @Positive
    public static final String ACCESSIBLE_TABLE_ROW_DESCRIPTION_CHANGED;

    @Positive
    public static final String ACCESSIBLE_TABLE_COLUMN_HEADER_CHANGED;

    @Positive
    public static final String ACCESSIBLE_TABLE_COLUMN_DESCRIPTION_CHANGED;

    @Positive
    public static final String ACCESSIBLE_ACTION_PROPERTY;

    @Positive
    public static final String ACCESSIBLE_HYPERTEXT_OFFSET;

    @Positive
    public static final String ACCESSIBLE_TEXT_PROPERTY;

    @Positive
    public static final String ACCESSIBLE_INVALIDATE_CHILDREN;

    @Positive
    public static final String ACCESSIBLE_TEXT_ATTRIBUTES_CHANGED;

    @Positive
    public static final String ACCESSIBLE_COMPONENT_BOUNDS_CHANGED;

    @Positive
    protected Accessible accessibleParent;

    @Positive
    protected String accessibleName;

    @Positive
    protected String accessibleDescription;

    @Positive
    public String getAccessibleName();

    @Positive
    @BeanProperty(preferred = true, description = "Sets the accessible name for the component.")
    @Positive
    public void setAccessibleName(String s);

    @Positive
    public String getAccessibleDescription();

    @Positive
    @BeanProperty(preferred = true, description = "Sets the accessible description for the component.")
    @Positive
    public void setAccessibleDescription(String s);

    @Positive
    public abstract AccessibleRole getAccessibleRole();

    @Positive
    public abstract AccessibleStateSet getAccessibleStateSet();

    @Positive
    public Accessible getAccessibleParent();

    @Positive
    public void setAccessibleParent(Accessible a);

    @Positive
    public abstract int getAccessibleIndexInParent();

    @Positive
    public abstract int getAccessibleChildrenCount();

    @Positive
    public abstract Accessible getAccessibleChild(int i);

    @Positive
    public abstract Locale getLocale() throws IllegalComponentStateException;

    @Positive
    public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public AccessibleAction getAccessibleAction();

    @Positive
    public AccessibleComponent getAccessibleComponent();

    @Positive
    public AccessibleSelection getAccessibleSelection();

    @Positive
    public AccessibleText getAccessibleText();

    @Positive
    public AccessibleEditableText getAccessibleEditableText();

    @Positive
    public AccessibleValue getAccessibleValue();

    @Positive
    public AccessibleIcon[] getAccessibleIcon();

    @Positive
    public AccessibleRelationSet getAccessibleRelationSet();

    @Positive
    public AccessibleTable getAccessibleTable();

    @Positive
    public void firePropertyChange(String propertyName, Object oldValue, Object newValue);
    @Positive
}

// CFWR semantic augmentation - variant 0
