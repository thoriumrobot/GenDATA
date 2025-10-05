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
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.ImageProducer;
import java.net.URL;
import java.security.AccessController;
import java.security.PrivilegedAction;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class SimpleBeanInfo implements BeanInfo {

    public SimpleBeanInfo() {
    }

    @Override
    public BeanDescriptor getBeanDescriptor();

    @Override
    public PropertyDescriptor[] getPropertyDescriptors();

    @Override
    public int getDefaultPropertyIndex();

    @Override
    public EventSetDescriptor[] getEventSetDescriptors();

    @Override
    public int getDefaultEventIndex();

    @Override
    public MethodDescriptor[] getMethodDescriptors();

    @Override
    public BeanInfo[] getAdditionalBeanInfo();

    @Override
    public Image getIcon(final int iconKind);

    public Image loadImage(final String resourceName);
}
