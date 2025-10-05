/*
    @Positive
 * Copyright (c) 1999, 2017, Oracle and/or its affiliates. All rights reserved.
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
@AnnotatedFor({ "interning" })
    @Positive
public class AccessibleRelation extends AccessibleBundle {

    @Positive
    @Interned
    @Positive
    public static final String LABEL_FOR;

    @Positive
    @Interned
    @Positive
    public static final String LABELED_BY;

    @Positive
    @Interned
    @Positive
    public static final String MEMBER_OF;

    @Positive
    @Interned
    @Positive
    public static final String CONTROLLER_FOR;

    @Positive
    @Interned
    @Positive
    public static final String CONTROLLED_BY;

    @Positive
    @Interned
    @Positive
    public static final String FLOWS_TO;

    @Positive
    @Interned
    @Positive
    public static final String FLOWS_FROM;

    @Positive
    @Interned
    @Positive
    public static final String SUBWINDOW_OF;

    @Positive
    @Interned
    @Positive
    public static final String PARENT_WINDOW_OF;

    @Positive
    @Interned
    @Positive
    public static final String EMBEDS;

    @Positive
    @Interned
    @Positive
    public static final String EMBEDDED_BY;

    @Positive
    @Interned
    @Positive
    public static final String CHILD_NODE_OF;

    @Positive
    @Interned
    @Positive
    public static final String LABEL_FOR_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String LABELED_BY_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String MEMBER_OF_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String CONTROLLER_FOR_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String CONTROLLED_BY_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String FLOWS_TO_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String FLOWS_FROM_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String SUBWINDOW_OF_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String PARENT_WINDOW_OF_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String EMBEDS_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String EMBEDDED_BY_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String CHILD_NODE_OF_PROPERTY;

    @Positive
    public AccessibleRelation(String key) {
    @Positive
    }

    @Positive
    public AccessibleRelation(String key, Object target) {
    @Positive
    }

    @Positive
    public AccessibleRelation(String key, Object[] target) {
    @Positive
    }

    @Positive
    public String getKey();

    @Positive
    public Object[] getTarget();

    @Positive
    public void setTarget(Object target);

    @Positive
    public void setTarget(Object[] target);
    @Positive
}

// CFWR semantic augmentation - variant 0
