/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2005, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.*;
    @Positive
import sun.awt.AppContext;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class ModalEventFilter implements EventFilter {

    @Positive
    protected Dialog modalDialog;

    @Positive
    protected boolean disabled;

    @Positive
    protected ModalEventFilter(Dialog modalDialog) {
    @Positive
    }

    @Positive
    Dialog getModalDialog();

    @Positive
    public FilterAction acceptEvent(AWTEvent event);

    @Positive
    protected abstract FilterAction acceptWindow(Window w);

    @Positive
    void disable();

    @Positive
    int compareTo(ModalEventFilter another);

    @Positive
    static ModalEventFilter createFilterForDialog(Dialog modalDialog);

    @Positive
    private static class ToolkitModalEventFilter extends ModalEventFilter {

    @Positive
        protected FilterAction acceptWindow(Window w);
    @Positive
    }

    @Positive
    private static class ApplicationModalEventFilter extends ModalEventFilter {

    @Positive
        protected FilterAction acceptWindow(Window w);
    @Positive
    }

    @Positive
    private static class DocumentModalEventFilter extends ModalEventFilter {

    @Positive
        protected FilterAction acceptWindow(Window w);
    @Positive
    }
    @Positive
}
