/*
    @Positive
 * Copyright (c) 2003, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.synth;

    @Positive
import org.checkerframework.checker.regex.qual.Regex;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.plaf.FontUIResource;
    @Positive
import java.awt.Font;
    @Positive
import java.util.*;
    @Positive
import java.util.regex.*;
    @Positive
import sun.swing.plaf.synth.*;
    @Positive
import sun.swing.BakedArrayList;

    @Positive
@AnnotatedFor({ "regex" })
    @Positive
class DefaultSynthStyleFactory extends SynthStyleFactory {

    @Positive
    public static final int NAME;

    @Positive
    public static final int REGION;

    @Positive
    public synchronized void addStyle(DefaultSynthStyle style, @Regex String path, int type) throws PatternSyntaxException;

    @Positive
    public synchronized SynthStyle getStyle(JComponent c, Region id);
    @Positive
}

// CFWR semantic augmentation - variant 1
