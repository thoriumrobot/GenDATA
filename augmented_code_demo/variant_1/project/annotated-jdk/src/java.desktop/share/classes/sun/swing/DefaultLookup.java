/*
    @Positive
 * Copyright (c) 2003, Oracle and/or its affiliates. All rights reserved.
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
package sun.swing;

    @Positive
import java.awt.Color;
    @Positive
import java.awt.Insets;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.border.Border;
    @Positive
import javax.swing.plaf.ComponentUI;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.awt.AppContext;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class DefaultLookup {

    @Positive
    public static void setDefaultLookup(@Nullable DefaultLookup lookup);

    @Positive
    @Nullable
    @Positive
    public static Object get(JComponent c, ComponentUI ui, String key);

    @Positive
    public static int getInt(JComponent c, ComponentUI ui, String key, int defaultValue);

    @Positive
    public static int getInt(JComponent c, ComponentUI ui, String key);

    @Positive
    @PolyNull
    @Positive
    public static Insets getInsets(JComponent c, ComponentUI ui, String key, @PolyNull Insets defaultValue);

    @Positive
    @Nullable
    @Positive
    public static Insets getInsets(JComponent c, ComponentUI ui, String key);

    @Positive
    public static boolean getBoolean(JComponent c, ComponentUI ui, String key, boolean defaultValue);

    @Positive
    public static boolean getBoolean(JComponent c, ComponentUI ui, String key);

    @Positive
    @PolyNull
    @Positive
    public static Color getColor(JComponent c, ComponentUI ui, String key, @PolyNull Color defaultValue);

    @Positive
    @Nullable
    @Positive
    public static Color getColor(JComponent c, ComponentUI ui, String key);

    @Positive
    @PolyNull
    @Positive
    public static Icon getIcon(JComponent c, ComponentUI ui, String key, @PolyNull Icon defaultValue);

    @Positive
    @Nullable
    @Positive
    public static Icon getIcon(JComponent c, ComponentUI ui, String key);

    @Positive
    @PolyNull
    @Positive
    public static Border getBorder(JComponent c, ComponentUI ui, String key, @PolyNull Border defaultValue);

    @Positive
    @Nullable
    @Positive
    public static Border getBorder(JComponent c, ComponentUI ui, String key);

    @Positive
    @Nullable
    @Positive
    public Object getDefault(JComponent c, ComponentUI ui, String key);
    @Positive
}

// CFWR semantic augmentation - variant 1
