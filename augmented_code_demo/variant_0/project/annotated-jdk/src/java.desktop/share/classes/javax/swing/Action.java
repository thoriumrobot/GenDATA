/*
    @Positive
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.beans.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public interface Action extends ActionListener {

    @Positive
    @Interned
    @Positive
    public static final String DEFAULT;

    @Positive
    @Interned
    @Positive
    public static final String NAME;

    @Positive
    @Interned
    @Positive
    public static final String SHORT_DESCRIPTION;

    @Positive
    @Interned
    @Positive
    public static final String LONG_DESCRIPTION;

    @Positive
    @Interned
    @Positive
    public static final String SMALL_ICON;

    @Positive
    @Interned
    @Positive
    public static final String ACTION_COMMAND_KEY;

    @Positive
    @Interned
    @Positive
    public static final String ACCELERATOR_KEY;

    @Positive
    @Interned
    @Positive
    public static final String MNEMONIC_KEY;

    @Positive
    @Interned
    @Positive
    public static final String SELECTED_KEY;

    @Positive
    public static final String DISPLAYED_MNEMONIC_INDEX_KEY;

    @Positive
    @Interned
    @Positive
    public static final String LARGE_ICON_KEY;

    @Positive
    public Object getValue(String key);

    @Positive
    public void putValue(String key, Object value);

    @Positive
    public void setEnabled(boolean b);

    @Positive
    public boolean isEnabled();

    @Positive
    default boolean accept(Object sender);

    @Positive
    public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public void removePropertyChangeListener(PropertyChangeListener listener);
    @Positive
}

// CFWR semantic augmentation - variant 0
