/*
    @Positive
 * Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
package sun.print;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.OutputStream;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Locale;
    @Positive
import javax.print.DocFlavor;
    @Positive
import javax.print.DocPrintJob;
    @Positive
import javax.print.StreamPrintService;
    @Positive
import javax.print.StreamPrintServiceFactory;
    @Positive
import javax.print.ServiceUIFactory;
    @Positive
import javax.print.attribute.Attribute;
    @Positive
import javax.print.attribute.AttributeSet;
    @Positive
import javax.print.attribute.AttributeSetUtilities;
    @Positive
import javax.print.attribute.HashAttributeSet;
    @Positive
import javax.print.attribute.HashPrintServiceAttributeSet;
    @Positive
import javax.print.attribute.PrintServiceAttribute;
    @Positive
import javax.print.attribute.PrintServiceAttributeSet;
    @Positive
import javax.print.attribute.Size2DSyntax;
    @Positive
import javax.print.event.PrintServiceAttributeListener;
    @Positive
import javax.print.attribute.standard.JobName;
    @Positive
import javax.print.attribute.standard.RequestingUserName;
    @Positive
import javax.print.attribute.standard.Chromaticity;
    @Positive
import javax.print.attribute.standard.ColorSupported;
    @Positive
import javax.print.attribute.standard.Copies;
    @Positive
import javax.print.attribute.standard.CopiesSupported;
    @Positive
import javax.print.attribute.standard.Fidelity;
    @Positive
import javax.print.attribute.standard.Media;
    @Positive
import javax.print.attribute.standard.MediaPrintableArea;
    @Positive
import javax.print.attribute.standard.MediaSize;
    @Positive
import javax.print.attribute.standard.MediaSizeName;
    @Positive
import javax.print.attribute.standard.OrientationRequested;
    @Positive
import javax.print.attribute.standard.PageRanges;
    @Positive
import javax.print.attribute.standard.SheetCollate;
    @Positive
import javax.print.attribute.standard.Sides;

    @Positive
public class PSStreamPrintService extends StreamPrintService implements SunPrinterJobService {

    @Positive
    public PSStreamPrintService(OutputStream out) {
    @Positive
    }

    @Positive
    public String getOutputFormat();

    @Positive
    public DocFlavor[] getSupportedDocFlavors();

    @Positive
    public DocPrintJob createPrintJob();

    @Positive
    public boolean usesClass(Class<?> c);

    @Positive
    public String getName();

    @Positive
    public void addPrintServiceAttributeListener(PrintServiceAttributeListener listener);

    @Positive
    public void removePrintServiceAttributeListener(PrintServiceAttributeListener listener);

    @Positive
    public <T extends PrintServiceAttribute> T getAttribute(Class<T> category);

    @Positive
    public PrintServiceAttributeSet getAttributes();

    @Positive
    public boolean isDocFlavorSupported(DocFlavor flavor);

    @Positive
    public Class<?>[] getSupportedAttributeCategories();

    @Positive
    public boolean isAttributeCategorySupported(Class<? extends Attribute> category);

    @Positive
    public Object getDefaultAttributeValue(Class<? extends Attribute> category);

    @Positive
    public Object getSupportedAttributeValues(Class<? extends Attribute> category, DocFlavor flavor, AttributeSet attributes);

    @Positive
    public boolean isAttributeValueSupported(Attribute attr, DocFlavor flavor, AttributeSet attributes);

    @Positive
    public AttributeSet getUnsupportedAttributes(DocFlavor flavor, AttributeSet attributes);

    @Positive
    public ServiceUIFactory getServiceUIFactory();

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 1
